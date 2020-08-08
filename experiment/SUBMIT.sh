# submit config
PYTHONPATH=.. python3 hydra_system.py
  -m \
  model.z_size=9 \
  framework=adavae,badavae \
  framework.cls.params.batch_logvar_mode='normal','reparameterize','all' \
