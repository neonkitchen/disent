import torch
from disent.frameworks.weaklysupervised.adavae import (AdaVae, estimate_shared)
from disent.frameworks.unsupervised.vae import bce_loss_with_logits, kl_normal_loss


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


class BoundedAdaVae(AdaVae):

    def compute_loss(self, batch, batch_idx):
        a_x, p_x, n_x = batch
        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parametrisation
        a_z_mean, a_z_logvar = self.model.encode_gaussian(a_x)
        p_z_mean, p_z_logvar = self.model.encode_gaussian(p_x)
        n_z_mean, n_z_logvar = self.model.encode_gaussian(n_x)
        a_z_logvar = self.mutate_z_logvar(a_z_logvar)
        p_z_logvar = self.mutate_z_logvar(p_z_logvar)
        n_z_logvar = self.mutate_z_logvar(n_z_logvar)
        # intercept and mutate z [SPECIFIC TO ADAVAE]
        (a_z_mean, a_z_logvar, p_z_mean, p_z_logvar), intercept_logs = self.intercept_z(a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar)
        # sample from latent distribution
        a_z_sampled = self.reparameterize(a_z_mean, a_z_logvar)
        p_z_sampled = self.reparameterize(p_z_mean, p_z_logvar)
        # reconstruct without the final activation
        a_x_recon = self.model.decode(a_z_sampled)
        p_x_recon = self.model.decode(p_z_sampled)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # reconstruction error
        a_recon_loss = bce_loss_with_logits(a_x, a_x_recon)  # E[log p(x|z)]
        p_recon_loss = bce_loss_with_logits(p_x, p_x_recon)  # E[log p(x|z)]
        ave_recon_loss = (a_recon_loss + p_recon_loss) / 2
        # KL divergence
        a_kl_loss = kl_normal_loss(a_z_mean, a_z_logvar)     # D_kl(q(z|x) || p(z|x))
        p_kl_loss = kl_normal_loss(p_z_mean, p_z_logvar)     # D_kl(q(z|x) || p(z|x))
        ave_kl_loss = (a_kl_loss + p_kl_loss) / 2
        # compute kl regularisation
        ave_kl_reg_loss = self.kl_regularization(ave_kl_loss)
        # compute combined loss - must be same as the BetaVAE
        loss = ave_recon_loss + ave_kl_reg_loss
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        return {
            'train_loss': loss,
            'recon_loss': ave_recon_loss,
            'kl_reg_loss': ave_kl_reg_loss,
            'kl_loss': ave_kl_loss,
            'elbo': -(ave_recon_loss + ave_kl_loss),
            **intercept_logs,
        }

    def intercept_z(self, a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar):
        # shared elements that need to be averaged, computed per pair in the batch.
        p_kl_deltas, p_kl_threshs, old_p_shared_mask = estimate_shared(a_z_mean, a_z_logvar, p_z_mean, p_z_logvar)
        n_kl_deltas, n_kl_threshs, old_n_shared_mask = estimate_shared(a_z_mean, a_z_logvar, n_z_mean, n_z_logvar)

        # modify threshold based on criterion and recompute if necessary
        # CORE of this approach!
        p_shared_mask, n_shared_mask = compute_constrained_masks(p_kl_deltas, old_p_shared_mask, n_kl_deltas, old_n_shared_mask)
        
        # make averaged variables
        new_args = self.make_averaged(a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, p_shared_mask)

        # return new args & generate logs
        return new_args, {
            'p_shared_before': old_p_shared_mask.sum(dim=1).float().mean(),
            'p_shared_after':      p_shared_mask.sum(dim=1).float().mean(),
            'n_shared_before': old_n_shared_mask.sum(dim=1).float().mean(),
            'n_shared_after':      n_shared_mask.sum(dim=1).float().mean(),
        }
    

# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def compute_constrained_masks(p_kl_deltas, p_shared_mask, n_kl_deltas, n_shared_mask):
    # number of changed factors
    p_shared_num = torch.sum(p_shared_mask, dim=1, keepdim=True)
    n_shared_num = torch.sum(n_shared_mask, dim=1, keepdim=True)

    # POSITIVE SHARED MASK
    # order from smallest to largest
    p_sort_indices = torch.argsort(p_kl_deltas, dim=1)
    # p_shared should be at least n_shared
    new_p_shared_num = torch.max(p_shared_num, n_shared_num)

    # NEGATIVE SHARED MASK
    # order from smallest to largest
    n_sort_indices = torch.argsort(n_kl_deltas, dim=1)
    # n_shared should be at most p_shared
    new_n_shared_num = torch.min(p_shared_num, n_shared_num)

    # COMPUTE NEW MASKS
    new_p_shared_mask = torch.zeros_like(p_shared_mask)
    new_n_shared_mask = torch.zeros_like(n_shared_mask)
    for i, (new_shared_p, new_shared_n) in enumerate(zip(new_p_shared_num, new_n_shared_num)):
        new_p_shared_mask[i, p_sort_indices[i, :new_shared_p]] = True
        new_n_shared_mask[i, n_sort_indices[i, :new_shared_n]] = True

    # return masks
    return new_p_shared_mask, new_n_shared_mask


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

