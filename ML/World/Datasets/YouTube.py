# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import os.path

import torch

from torch.utils.data import Dataset

from World.Memory import Mem


class YouTube(Dataset):
    """
    YouTube videos
    """

    def __init__(self, url=None, file=None, _root_=None):  # TODO Eventually, temporal outputs
        from vidgear.gears import CamGear

        assert url or file, 'URL or URL-file parameter are required.'

        urls = []
        if file is not None:  # Can pass in file of URLs
            with open(file, 'r') as f:
                urls = list(f.readlines())
        if url is not None:  # And/or just a YouTube URL
            urls.append(url)

        # TODO Don't download the same URL twice
        for url in urls:  # Get frames  Note: videos get stored on hard disk in full resolution
            i = 0

            frames = []
            self.frames = []

            print('Downloading YouTube video from URL:', url)

            path = _root_ + f'Download_URL_{url.rsplit("/")[-1]}'

            # Load pre-existing download
            if os.path.isfile(f'{path}/done'):
                print(f'Located pre-existing download: {path}')

                for mmap in os.listdir(path):
                    if 'done' not in mmap.rsplit('/')[-1]:
                        mem = Mem(None, path=path + '/' + mmap).load()
                        self.frames += list(enumerate([mem] * len(mem)))
                break

            video = CamGear(source=url, stream_mode=True).start()

            while True:
                frame = video.read()

                # Save in chunks of 256 on hard disk
                if len(frames) and len(frames) % 256 == 0 or frame is None:
                    mem = Mem(torch.stack(frames), f'{path}/{i}').mmap()
                    self.frames += list(enumerate([mem] * len(frames)))
                    frames = []
                    i += 1

                if frame is None:
                    open(f'{path}/done', 'w').close()
                    break

                frames.append(torch.as_tensor(frame, dtype=torch.float32).permute(2, 0, 1))

    def __getitem__(self, ind):
        ind, mem = self.frames[ind]
        return mem[ind]

    def __len__(self):
        return len(self.frames)

