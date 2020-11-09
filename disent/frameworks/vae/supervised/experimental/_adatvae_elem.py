import torch
import numpy as np
from disent.frameworks.vae.supervised.experimental._adatvae import AdaTripletVae, blend

import logging

log = logging.getLogger(__name__)


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


def elem_triplet_loss(anc, pos, neg, margin=.1, p=1):
    assert p == 1, 'Element-wise triplet only supports p==1'
    return dist_elem_triplet_loss(anc - pos, anc - neg, margin=margin)


def dist_elem_triplet_loss(pos_delta, neg_delta, margin=1., p=1):
    assert p == 1, 'Element-wise triplet only supports p==1'
    p_dist = torch.abs(pos_delta)
    n_dist = torch.abs(neg_delta)
    loss = torch.clamp_min(p_dist - n_dist + margin, 0)
    return loss.mean()


class AdaTripletVaeElementWise(AdaTripletVae):

    def __init__(self, *args, **kwargs):
        assert kwargs['triplet_p'] == 1, 'Element Wise Only Supports p==1'
        super().__init__(*args, **kwargs)

        # Override!
        self.triplet_loss = elem_triplet_loss
        self.dist_triplet_loss = dist_elem_triplet_loss


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
