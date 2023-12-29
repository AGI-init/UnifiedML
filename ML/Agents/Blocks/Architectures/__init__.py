# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from Agents.Blocks.Architectures.MLP import MLP, Dense

from Agents.Blocks.Architectures.Vision.CNN import CNN, Conv

from Agents.Blocks.Architectures.Residual import Residual

from Agents.Blocks.Architectures.Vision.ResNet import MiniResNet, MiniResNet as ResNet, ResNet18, ResNet50
from Agents.Blocks.Architectures.Vision.ConvMixer import ConvMixer
from Agents.Blocks.Architectures.Vision.ConvNeXt import ConvNeXt, ConvNeXtTiny, ConvNeXtBase

from Agents.Blocks.Architectures.MultiHeadAttention import Attention, MHDPA, CrossAttention, SelfAttention, ReLA

from Agents.Blocks.Architectures.Transformer import AttentionBlock, CrossAttentionBlock, SelfAttentionBlock, Transformer

from Agents.Blocks.Architectures.Vision.ViT import ViT, ViT as VisionTransformer
from Agents.Blocks.Architectures.Vision.CoAtNet import CoAtNet, CoAtNet0, CoAtNet1, CoAtNet2, CoAtNet3, CoAtNet4

from Agents.Blocks.Architectures.Vision import DCGAN

from Agents.Blocks.Architectures.Perceiver import Perceiver

from Agents.Blocks.Architectures.RN import RN, RN as RelationNetwork

from Agents.Blocks.Architectures.Vision.CNN import AvgPool
from Agents.Blocks.Architectures.Vision.ViT import CLSPool

from Agents.Blocks.Architectures.Vision.TIMM import TIMM
