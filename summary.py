import argparse
import numpy as np
import os
import tabulate
import torch
import torch.nn.functional as F

import data
import models
import curves
import utils

from torchinfo import summary

parser = argparse.ArgumentParser(description='Prints model summary')
parser.add_argument('--dir', type=str, default='/tmp/plane', metavar='DIR',
                    help='training directory (default: /tmp/plane)')
parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')

parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')

args = parser.parse_args()

architecture = getattr(models, args.model)

num_classes = 10

if args.curve is not None:
    curve = getattr(curves, args.curve)

    model = curves.CurveNet(
        num_classes,
        curve,
        architecture.curve,
        args.num_bends,
        architecture_kwargs=architecture.kwargs,
    )
    model.cuda()

    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint['model_state'])

else:
    model = architecture.base(num_classes=num_classes, **architecture.kwargs)
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint['model_state'])

summary(model)

if args.curve is not None:
    # w = model.weightsList(0.5)
    #
    # for param in w:
    #     print(torch.tensor(param).shape)
    #
    # print("end")
    #
    # modelPoint = architecture.base(num_classes=num_classes, **architecture.kwargs)
    # for i, param_cur in enumerate(modelPoint.parameters()):
    #     print(param_cur.shape)
    #     param_cur.data = w[i]

    summary(model.modelAt(0.5, architecture))
