# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build as build1
from .detr_pretrained import build


def build_model(args):
    if args.pretrained is None:
        return build1(args)
    else:
        return build(args)
