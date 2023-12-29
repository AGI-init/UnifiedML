# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import os

import torch
from torch import nn

from Utils import gather


"""
Huge thank you to https://github.com/IDEA-Research/GroundingDINO/issues/102
"""
def predict_batch(model, image, caption, device):
    from groundingdino.util.inference import preprocess_caption

    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image, captions=[caption for _ in range(len(image))])  # [batch_size, 900, 4]

    return outputs["pred_boxes"], outputs["pred_logits"]


class GroundingDINO(nn.Module):
    """
    GroundingDINO - one of the coolest vision-language models, combining DINO with language.
    Repo: ShilongLiu/GroundingDINO.
    """

    def __init__(self, caption='little robot dog', device='cpu'):
        super().__init__()

        try:
            from groundingdino.models import build_model
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import clean_state_dict

            from huggingface_hub import hf_hub_download
        except Exception as e:
            print(e)
            raise RuntimeError('Make sure you have installed GroundingDINO: \n'
                               '$ pip install groundingdino-py\n'
                               'and huggingface_hub. See ShilongLiu/GroundingDINO.')

        def load_model_hf(model_config_path, repo_id, filename):  # TODO device?
            args = SLConfig.fromfile(model_config_path)
            model = build_model(args)
            args.device = device

            cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
            checkpoint = torch.load(cache_file, map_location=device)
            model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
            _ = model.eval()
            return model

        self.caption = caption

        repo_id = 'ShilongLiu/GroundingDINO'

        cache_config_file = hf_hub_download(repo_id=repo_id, filename='GroundingDINO_SwinB.cfg.py')

        self.GroundingDINO = load_model_hf(cache_config_file,
                                           repo_id=repo_id,
                                           filename='groundingdino_swinb_cogcoor.pth').to(device)

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def forward(self, obs, caption=None):
        boxes, logits = self.predict_batch(
            image=obs.to(torch.float32),
            caption=caption or self.caption,
            device=obs.device
        )

        return boxes, logits

    def predict_batch(self, image, caption, device):
        from groundingdino.util.inference import preprocess_caption

        caption = preprocess_caption(caption=caption)

        image = image.to(device)

        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        with torch.no_grad():
            outputs = self.GroundingDINO(image, captions=[caption for _ in range(len(image))])  # [batch_size, 900, 4]

        logits, _ = outputs["pred_logits"].max(2)
        logits, ind = logits.unsqueeze(-1).max(1)
        boxes = gather(outputs["pred_boxes"], ind, 1).squeeze(1)

        return boxes, logits


# Example Usage:
# python Run.py env=YouTube env.url='https://youtube.com/live/...=share'
#                           env.transform=Sequential
#                           env.transform._targets_='["transforms.Resize(32)","World.Environments.YouTube.AutoLabel"]'

class AutoLabel(nn.Module):
    """
    A augmentation for auto-labelling object locations based on image and caption
    """

    def __init__(self, caption='little robot dog', device=None):
        super().__init__()

        # SotA object detection foundation model
        self.GroundingDINO = GroundingDINO(caption, device)

        mps = getattr(torch.backends, 'mps', None)  # M1 MacBook speedup

        # Set device
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() \
            else 'mps' if mps and mps.is_available() \
            else 'cpu'

        self._exp_ = True  # Flag that toggles passing in full exp/batch-dict instead of just obs

    def forward(self, exp):
        if self.device is not None:
            exp.obs = exp.obs.to(self.device)

        boxes, logits = self.GroundingDINO(exp.obs)

        exp.label = boxes

        return exp


# TODO Custom metric or transform that just saves image and action pair. If metric, can do MP4. Maybe vlogger.


# TODO Delete
# if __name__ == '__main__':
#     import os
#
#     import numpy as np
#     from PIL import Image
#     import cv2
#
#     from heic2png import HEIC2PNG  # pip install HEIC2PNG
#
#     import groundingdino.datasets.transforms as T
#     # import torchvision.transforms as T
#     from groundingdino.util.inference import annotate
#
#
#     def load_image(image_path):
#         transform = T.Compose(
#             [
#                 T.RandomResize([800], max_size=1333),
#                 T.ToTensor(),
#                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ]
#         )
#         src = Image.open(image_path).convert("RGB")
#         img = np.asarray(src)
#         image_transformed, _ = transform(src, None)
#         return img, image_transformed
#
#
#     path = '/Users/sam/Code/Playground/Magic/images/'
#
#     images = []
#     sources = []
#
#     for i, local_image_path in enumerate(os.listdir(path)):
#         if local_image_path[-5:] == '.heic':
#             HEIC2PNG(path + local_image_path).save()
#             os.system(f'rm {path}{local_image_path}')
#             local_image_path = local_image_path.replace('.heic', '.png')
#
#         image_source, image = load_image(path + local_image_path)
#         sources.append(image_source)
#         image = torch.as_tensor(image)
#         if images and image.shape != images[0].shape:
#             image = image.permute(0, 2, 1)
#         images.append(image)
#         if i > 1:
#             break
#
#     images = torch.stack(images)
#     print(images.shape)
#     GD = GroundingDINO()
#     boxes, logits = GD(images)
#
#     print(boxes.shape, logits.shape)
#
#     # indices = logits.argmax(1)
#     # boxes = gather(boxes, indices, 1)  # Highest proba bounding-box
#     #
#     # print(boxes.shape)
#
#     for i, image in enumerate(sources):
#         annotated_frame = annotate(image_source=image, boxes=boxes[i].unsqueeze(0),
#                                    logits=logits[i], phrases=['little robot dog'])
#
#         os.makedirs('./Image', exist_ok=True)
#
#         cv2.imwrite('./Image/' + str(i) + '.png', annotated_frame)
