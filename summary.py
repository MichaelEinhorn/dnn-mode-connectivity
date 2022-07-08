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


def flatten(el):
    flattened = [flatten(children) for children in el.children()]
    res = [el]
    for c in flattened:
        res += c
    return res

def main():
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

        if args.ckpt is not None:
            checkpoint = torch.load(args.ckpt)
            model.load_state_dict(checkpoint['model_state'])

    else:
        model = architecture.base(num_classes=num_classes, **architecture.kwargs)

        if args.ckpt is not None:
            checkpoint = torch.load(args.ckpt)
            model.load_state_dict(checkpoint['model_state'])

    summary(model, input_size=(128, 3, 32, 32))


if __name__ == '__main__':
    main()
