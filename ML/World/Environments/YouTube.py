# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch


class YouTube:
    """
    Live-streaming environment
    """

    def __init__(self, url, train=True, steps=1000):
        from vidgear.gears import CamGear

        url = url.split('?feature')[0]

        self.video = CamGear(source=url, stream_mode=True, logging=True).start()

        self.train = train
        self.steps = steps  # Controls evaluation episode length
        self.episode_step = 0

    def step(self, action):
        return self.reset()

    def reset(self):
        self.episode_step += 1
        return {'obs': torch.as_tensor(self.video.read()).permute(2, 0, 1).unsqueeze(0),
                'done': not self.train and not self.episode_step % self.steps}
