#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

import dataclasses
import logging
import os

import hydra
import pytorch_lightning as pl
import torch
import torch.utils.data
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.loggers import LoggerCollection
from pytorch_lightning.loggers import WandbLogger

from disent import metrics
from disent.frameworks.framework import BaseFramework
from disent.model.ae.base import AutoEncoder
from disent.model.init import init_model_weights
from disent.util import DisentConfigurable
from disent.util import make_box_str
from experiment.util.callbacks import LoggerProgressCallback
from experiment.util.callbacks import VaeDisentanglementLoggingCallback
from experiment.util.callbacks import VaeLatentCycleLoggingCallback
from experiment.util.callbacks.callbacks_vae import VaeLatentCorrelationLoggingCallback
from experiment.util.hydra_data import HydraDataModule
from experiment.util.hydra_utils import instantiate_recursive
from experiment.util.hydra_utils import make_non_strict
from experiment.util.hydra_utils import merge_specializations
from experiment.util.run_utils import log_error_and_exit
from experiment.util.run_utils import set_debug_logger
from experiment.util.run_utils import set_debug_trainer


log = logging.getLogger(__name__)


# ========================================================================= #
# HYDRA CONFIG HELPERS                                                      #
# ========================================================================= #


def hydra_check_cuda(cfg):
    # set cuda
    if cfg.trainer.cuda in {'try_cuda', None}:
        cfg.trainer.cuda = torch.cuda.is_available()
        if not cfg.trainer.cuda:
            log.warning('CUDA was requested, but not found on this system... CUDA has been disabled!')
    else:
        if not torch.cuda.is_available():
            if cfg.trainer.cuda:
                log.error('trainer.cuda=True but CUDA is not available on this machine!')
                raise RuntimeError('CUDA not available!')
            else:
                log.warning('CUDA is not available on this machine!')
        else:
            if not cfg.trainer.cuda:
                log.warning('CUDA is available but is not being used!')


def hydra_check_datadir(prepare_data_per_node, cfg):
    if not os.path.isabs(cfg.dataset.data_dir):
        log.warning(
            f'A relative path was specified for dataset.data_dir={repr(cfg.dataset.data_dir)}.'
            f' This is probably an error! Using relative paths can have unintended consequences'
            f' and performance drawbacks if the current working directory is on a shared/network drive.'
            f' Hydra config also uses a new working directory for each run of the program, meaning'
            f' data will be repeatedly downloaded.'
        )
        if prepare_data_per_node:
            log.error(
                f'trainer.prepare_data_per_node={repr(prepare_data_per_node)} but  dataset.data_dir='
                f'{repr(cfg.dataset.data_dir)} is a relative path which may be an error! Try specifying an'
                f' absolute path that is guaranteed to be unique from each node, eg. dataset.data_dir=/tmp/dataset'
            )
        raise RuntimeError('dataset.data_dir={repr(cfg.dataset.data_dir)} is a relative path!')


def hydra_make_logger(cfg):
    loggers = []

    # initialise logging dict
    cfg.setdefault('logging', {})

    if ('wandb' in cfg.logging) and cfg.logging.wandb.setdefault('enabled', True):
        loggers.append(WandbLogger(
            offline=cfg.logging.wandb.setdefault('offline', False),
            entity=cfg.logging.wandb.setdefault('entity', None),  # cometml: workspace
            project=cfg.logging.wandb.project,                    # cometml: project_name
            name=cfg.logging.wandb.name,                          # cometml: experiment_name
            group=cfg.logging.wandb.setdefault('group', None),    # experiment group
            tags=cfg.logging.wandb.setdefault('tags', None),      # experiment tags
            save_dir=hydra.utils.to_absolute_path(cfg.logging.logs_dir),  # relative to hydra's original cwd
        ))
    else:
        cfg.logging.setdefault('wandb', dict(enabled=False))

    if ('cometml' in cfg.logging) and cfg.logging.cometml.setdefault('enabled', True):
        loggers.append(CometLogger(
            offline=cfg.logging.cometml.setdefault('offline', False),
            workspace=cfg.logging.cometml.setdefault('workspace', None),  # wandb: entity
            project_name=cfg.logging.cometml.project,                     # wandb: project
            experiment_name=cfg.logging.cometml.name,                     # wandb: name
            api_key=os.environ['COMET_API_KEY'],                          # TODO: use dotenv
            save_dir=hydra.utils.to_absolute_path(cfg.logging.logs_dir),  # relative to hydra's original cwd
        ))
    else:
        cfg.logging.setdefault('cometml', dict(enabled=False))

    return LoggerCollection(loggers) if loggers else None  # lists are turned into a LoggerCollection by pl


def hydra_append_progress_callback(callbacks, cfg):
    if 'progress' in cfg.callbacks:
        callbacks.append(LoggerProgressCallback(
            interval=cfg.callbacks.progress.interval
        ))


def hydra_append_latent_cycle_logger_callback(callbacks, cfg):
    if 'latent_cycle' in cfg.callbacks:
        if cfg.logging.wandb.enabled:
            # this currently only supports WANDB logger
            callbacks.append(VaeLatentCycleLoggingCallback(
                seed=cfg.callbacks.latent_cycle.seed,
                every_n_steps=cfg.callbacks.latent_cycle.every_n_steps,
                begin_first_step=False,
                mode=cfg.callbacks.latent_cycle.mode,
            ))
        else:
            log.warning('latent_cycle callback is not being used because wandb is not enabled!')


def hydra_append_metric_callback(callbacks, cfg):
    # set default values used later
    default_every_n_steps = cfg.metrics.setdefault('default_every_n_steps', 3600)
    default_on_final = cfg.metrics.setdefault('default_on_final', True)
    default_on_train = cfg.metrics.setdefault('default_on_train', True)
    # get metrics
    metric_list = cfg.metrics.setdefault('metric_list', [])
    if metric_list == 'all':
        cfg.metrics.metric_list = metric_list = [{k: {}} for k in metrics.DEFAULT_METRICS]
    # get metrics
    new_metrics_list = []
    for i, metric in enumerate(metric_list):
        # fix the values
        if isinstance(metric, str):
            metric = {metric: {}}
        ((name, settings),) = metric.items()
        if settings is None:
            settings = {}
        new_metrics_list.append({name: settings})
        # get metrics
        every_n_steps = settings.get('every_n_steps', default_every_n_steps)
        train_metric = [metrics.FAST_METRICS[name]] if settings.get('on_train', default_on_train) else None
        final_metric = [metrics.DEFAULT_METRICS[name]] if settings.get('on_final', default_on_final) else None
        # add the metric callback
        if final_metric or train_metric:
            callbacks.append(VaeDisentanglementLoggingCallback(
                every_n_steps=every_n_steps,
                step_end_metrics=train_metric,
                train_end_metrics=final_metric,
            ))
    cfg.metrics.metric_list = new_metrics_list


def hydra_append_correlation_callback(callbacks, cfg):
    if 'correlation' in cfg.callbacks:
        callbacks.append(VaeLatentCorrelationLoggingCallback(
            repeats_per_factor=cfg.callbacks.correlation.repeats_per_factor,
            every_n_steps=cfg.callbacks.correlation.every_n_steps,
            begin_first_step=False,
        ))


def hydra_register_schedules(module: BaseFramework, cfg):
    if cfg.schedules is None:
        cfg.schedules = {}
    if cfg.schedules:
        log.info(f'Registering Schedules:')
        for target, schedule in cfg.schedules.items():
            module.register_schedule(target, instantiate_recursive(schedule), logging=True)


def hydra_create_framework_config(cfg):
    # create framework config - this is also kinda hacky
    framework_cfg: DisentConfigurable.cfg = hydra.utils.instantiate({
        **cfg.framework.module,
        **dict(_target_=cfg.framework.module._target_ + '.cfg')
    })
    # warn if some of the cfg variables were not overridden
    missing_keys = sorted(set(framework_cfg.get_keys()) - set(cfg.framework.module.keys()))
    if missing_keys:
        log.error(f'Framework {repr(cfg.framework.name)} is missing config keys for:')
        for k in missing_keys:
            log.error(f'{repr(k)}')
    # update config params in case we missed variables in the cfg
    cfg.framework.module.update(framework_cfg.to_dict())
    # return config
    return framework_cfg


def hydra_create_framework(framework_cfg, datamodule, cfg):
    return hydra.utils.instantiate(
        dict(_target_=cfg.framework.module._target_),
        make_optimizer_fn=lambda params: hydra.utils.instantiate(cfg.optimizer.cls, params),
        make_model_fn=lambda: init_model_weights(
            AutoEncoder(
                encoder=hydra.utils.instantiate(cfg.model.encoder),
                decoder=hydra.utils.instantiate(cfg.model.decoder)
            ), mode=cfg.model.weight_init
        ),
        # apply augmentations to batch on GPU which can be faster than via the dataloader
        batch_augment=datamodule.batch_augment,
        cfg=framework_cfg
    )


# ========================================================================= #
# RUNNER                                                                    #
# ========================================================================= #


def run(cfg: DictConfig):
    cfg = make_non_strict(cfg)

    # -~-~-~-~-~-~-~-~-~-~-~-~- #
    # INITIALISE & SETDEFAULT IN CONFIG
    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    # create trainer loggers & callbacks & initialise error messages
    logger = set_debug_logger(hydra_make_logger(cfg))

    # print useful info
    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    # hydra config does not support variables in defaults lists, we handle this manually
    cfg = merge_specializations(cfg, CONFIG_PATH, run)

    # check CUDA setting
    cfg.trainer.setdefault('cuda', 'try_cuda')
    hydra_check_cuda(cfg)
    # check data preparation
    prepare_data_per_node = cfg.trainer.setdefault('prepare_data_per_node', True)
    hydra_check_datadir(prepare_data_per_node, cfg)

    # TRAINER CALLBACKS
    callbacks = []
    hydra_append_progress_callback(callbacks, cfg)
    hydra_append_latent_cycle_logger_callback(callbacks, cfg)
    hydra_append_metric_callback(callbacks, cfg)
    hydra_append_correlation_callback(callbacks, cfg)

    # HYDRA MODULES
    datamodule = HydraDataModule(cfg)
    framework_cfg = hydra_create_framework_config(cfg)
    framework = hydra_create_framework(framework_cfg, datamodule, cfg)

    # register schedules
    hydra_register_schedules(framework, cfg)

    # Setup Trainer
    trainer = set_debug_trainer(pl.Trainer(
        log_every_n_steps=cfg.logging.setdefault('log_every_n_steps', 50),
        flush_logs_every_n_steps=cfg.logging.setdefault('flush_logs_every_n_steps', 100),
        logger=logger,
        callbacks=callbacks,
        gpus=1 if cfg.trainer.cuda else 0,
        max_epochs=cfg.trainer.setdefault('epochs', 100),
        max_steps=cfg.trainer.setdefault('steps', None),
        prepare_data_per_node=prepare_data_per_node,
        progress_bar_refresh_rate=0,  # ptl 0.9
        terminate_on_nan=True,  # we do this here so we don't run the final metrics
        # TODO: re-enable this in future... something is not compatible
        #       with saving/checkpointing models + allow enabling from the
        #       config. Seems like something cannot be pickled?
        checkpoint_callback=False,
    ))

    # -~-~-~-~-~-~-~-~-~-~-~-~- #
    # BEGIN TRAINING
    # -~-~-~-~-~-~-~-~-~-~-~-~- #

    # print the config
    log.info(f'Final Config Is:\n{make_box_str(OmegaConf.to_yaml(cfg))}')

    # save hparams TODO: I think this is a pytorch lightning bug... The trainer should automatically save these if hparams is set.
    framework.hparams = cfg
    if trainer.logger:
        trainer.logger.log_hyperparams(framework.hparams)

    # fit the model
    # -- if an error/signal occurs while pytorch lightning is
    #    initialising the training process we cannot capture it!
    trainer.fit(framework, datamodule=datamodule)


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config'))
    CONFIG_NAME = 'config'

    @hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
    def main(cfg: DictConfig):
        try:
            run(cfg)
        except Exception as e:
            log_error_and_exit(err_type='experiment error', err_msg=str(e))

    try:
        main()
    except KeyboardInterrupt as e:
        log_error_and_exit(err_type='interrupted', err_msg=str(e), exc_info=False)
    except Exception as e:
        log_error_and_exit(err_type='hydra error', err_msg=str(e))


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
