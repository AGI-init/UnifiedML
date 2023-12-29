# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from pathlib import Path
import imageio  # M1 Mac: Additionally run: conda install imageio imageio-ffmpeg

import torch
from torchvision.utils import save_image


class Vlogger:
    def __init__(self, fps, path='.', reel=False):
        self.save_path = Path(path)
        self.save_path.mkdir(exist_ok=True, parents=True)
        self.fps = fps

        # Saves image reels instead of video
        self.reel = reel

    def dump(self, vlog, name="Video_Image"):
        if self.reel:
            c, h, w = (min(vlog[0].shape[-3], 3),  # Undoing frame-stack if necessary (max = 3 channels per image)
                       vlog[0].shape[-2], vlog[0].shape[-1])  # TODO graphs, spectrogram for 1d
            vlogs = list(map(torch.as_tensor, vlog))
            save_image(torch.stack(vlogs).view(-1, c, h, w), str(self.save_path / (name + '.png')))
        else:
            # Assumes channel-last format
            imageio.mimsave(str(self.save_path / (name + '.mp4')), vlog, fps=self.fps)
