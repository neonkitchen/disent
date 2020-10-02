from disent.frameworks.vae.loss import bce_loss_with_logits, kl_normal_loss
from disent.frameworks.vae.supervised._tgadavae import triplet_loss
from disent.frameworks.vae.unsupervised import BetaVae
import torch.nn.functional as F

# ========================================================================= #
# tbadavae                                                                  #
# ========================================================================= #


class TripletVae(BetaVae):

    def __init__(
            self,
            make_optimizer_fn,
            make_model_fn,
            batch_augment=None,
            beta=4,
            triplet_margin=0.1,
            triplet_scale=1,
    ):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, beta=beta)
        self.triplet_margin = triplet_margin
        self.triplet_scale = triplet_scale

    def compute_training_loss(self, batch, batch_idx):
        (a_x, p_x, n_x), (a_x_targ, p_x_targ, n_x_targ) = batch['x'], batch['x_targ']

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parametrisation
        a_z_mean, a_z_logvar = self.encode_gaussian(a_x)
        p_z_mean, p_z_logvar = self.encode_gaussian(p_x)
        n_z_mean, n_z_logvar = self.encode_gaussian(n_x)
        # sample from latent distribution
        a_z_sampled = self.reparameterize(a_z_mean, a_z_logvar)
        p_z_sampled = self.reparameterize(p_z_mean, p_z_logvar)
        n_z_sampled = self.reparameterize(n_z_mean, n_z_logvar)
        # reconstruct without the final activation
        a_x_recon = self.decode_partial(a_z_sampled)
        p_x_recon = self.decode_partial(p_z_sampled)
        n_x_recon = self.decode_partial(n_z_sampled)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # reconstruction error
        a_recon_loss = bce_loss_with_logits(a_x_recon, a_x_targ)  # E[log p(x|z)]
        p_recon_loss = bce_loss_with_logits(p_x_recon, p_x_targ)  # E[log p(x|z)]
        n_recon_loss = bce_loss_with_logits(n_x_recon, n_x_targ)  # E[log p(x|z)]
        ave_recon_loss = (a_recon_loss + p_recon_loss + n_recon_loss) / 3
        # KL divergence
        a_kl_loss = kl_normal_loss(a_z_mean, a_z_logvar)  # D_kl(q(z|x) || p(z|x))
        p_kl_loss = kl_normal_loss(p_z_mean, p_z_logvar)  # D_kl(q(z|x) || p(z|x))
        n_kl_loss = kl_normal_loss(n_z_mean, n_z_logvar)  # D_kl(q(z|x) || p(z|x))
        ave_kl_loss = (a_kl_loss + p_kl_loss + n_kl_loss) / 3
        # compute kl regularisation
        ave_kl_reg_loss = self.kl_regularization(ave_kl_loss)
        # augment loss (0 for this)
        augment_loss, augment_loss_logs = self.augment_loss(a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar)
        # compute combined loss - must be same as the BetaVAE
        loss = ave_recon_loss + ave_kl_reg_loss + augment_loss
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        return {
            'train_loss': loss,
            'recon_loss': ave_recon_loss,
            'kl_reg_loss': ave_kl_reg_loss,
            'kl_loss': ave_kl_loss,
            'elbo': -(ave_recon_loss + ave_kl_loss),
            **augment_loss_logs,
        }

    def augment_loss(self, a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar):
        loss_triplet = triplet_loss(a_z_mean, p_z_mean, n_z_mean, margin=self.triplet_margin)
        augmented_loss = self.triplet_scale * loss_triplet
        return augmented_loss, {
            'triplet_loss': loss_triplet,
            'triplet_loss_torch': F.triplet_margin_loss(a_z_mean, p_z_mean, n_z_mean, margin=self.triplet_margin)
        }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #