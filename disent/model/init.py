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

import logging

from torch import nn

from disent.util import colors as c


log = logging.getLogger(__name__)


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


def init_model_weights(model: nn.Module, mode='xavier_normal'):
    count = 0

    # get default mode
    if mode is None:
        mode = 'default'

    def init_normal(m):
        nonlocal count
        init, count = False, count + 1

        # actually initialise!
        if mode == 'xavier_normal':
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                init = True
        elif mode == 'default':
            pass
        else:
            raise KeyError(f'Unknown init mode: {repr(mode)}')

        # print messages
        if init:
            log.info(f'| {count:03d} {c.lGRN}INIT{c.RST}: {m.__class__.__name__}')
        else:
            log.info(f'| {count:03d} {c.lRED}SKIP{c.RST}: {m.__class__.__name__}')

    log.info(f'Initialising Model Layers: {mode}')
    model.apply(init_normal)

    return model


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
