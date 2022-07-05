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
        model.cuda()

        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['model_state'])

        t = 0.2
        layerList = []
        print("printing layers")
        i = 0
        coeffs_t = model.coeff_layer(t)



        # for layer in flatten(model):
        #     # print(i)
        #     i += 1
        #     # print(layer)
        #     if not hasattr(layer, "makeLayer"):
        #         if not isinstance(layer, torch.nn.ModuleList) and not isinstance(layer, models.BasicBlockCurve) \
        #                 and not isinstance(layer, curves.CurveNet) and not isinstance(layer, models.PreResNetCurve) \
        #                 and not isinstance(layer, curves.Bezier):
        #             layerList.append(layer)
        #     else:
        #         layerList.append(layer.makeLayer(coeffs_t))


        # for layer in model.children():
        #     # print(layer)
        #     if hasattr(layer, "children"):
        #         for layer2 in layer.children():
        #             if hasattr(layer2, "children"):
        #                 for layer3 in layer2.children():
        #                     if hasattr(layer3, "children"):
        #                         for layer4 in layer3.children():
        #                             print(i)
        #                             i += 1
        #                             print(layer4)
        #                             if not hasattr(layer4, "makeLayer"):
        #                                 layerList.append(layer4)
        #                             else:
        #                                 layerList.append(layer4.makeLayer(coeffs_t))
        #                     else:
        #                         print(i)
        #                         i += 1
        #                         print(layer3)
        #                         if not hasattr(layer3, "makeLayer"):
        #                             layerList.append(layer3)
        #                         else:
        #                             layerList.append(layer3.makeLayer(coeffs_t))
        #             else:
        #                 print(i)
        #                 i += 1
        #                 print(layer2)
        #                 if not hasattr(layer2, "makeLayer"):
        #                     layerList.append(layer2)
        #                 else:
        #                     layerList.append(layer2.makeLayer(coeffs_t))
        #     else:
        #         print(i)
        #         i += 1
        #         print(layer)
        #         if not hasattr(layer, "makeLayer"):
        #             layerList.append(layer)
        #         else:
        #             layerList.append(layer.makeLayer(coeffs_t))




        # layerList2 = []
        # base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
        # # checkpoint = torch.load("history/model1/checkpoint-200.pt")
        # # base_model.load_state_dict(checkpoint['model_state'])
        #
        # p = model.weights(t)
        # offset = 0
        # for parameter in base_model.parameters():
        #     size = np.prod(parameter.size())
        #     value = p[offset:offset + size].reshape(parameter.size())
        #     parameter.data.copy_(torch.from_numpy(value))
        #     offset += size

        base_model = model.modelAt(t, architecture)

        # i=0
        # for layer in flatten(modelPoint):
        #     # print(i)
        #     i += 1
        #     # print(layer)
        #     if not isinstance(layer, torch.nn.ModuleList) and not isinstance(layer, models.BasicBlock) \
        #             and not isinstance(layer, torch.nn.Sequential) and not isinstance(layer, models.PreResNetBase) \
        #             and not isinstance(layer, curves.Bezier):
        #         layerList2.append(layer)

        # print(layerList, len(layerList))
        #
        # print(layerList2, len(layerList2))
        #
        # print("zip")
        # for layer1, layer2 in zip(layerList, layerList2):
        #     # print(layer1)
        #     # print(layer2)
        #     # print("\n")
        #
        #     if hasattr(layer1, "weight"):
        #         weight = layer1.weight
        #         if weight is not None:
        #             layer2.weight.data.copy_(weight)
        #     if hasattr(layer1, "bias"):
        #         bias = layer1.bias
        #         if bias is not None:
        #             layer2.bias.data.copy_(bias)



        loaders, num_classes = data.loaders(
            "CIFAR10",
            "CIFAR",
            128,
            2,
            "ResNet",
            False,
            shuffle_train=False
        )
        criterion = F.cross_entropy
        base_model.cuda()

        utils.update_bn(loaders['train'], base_model)
        utils.update_bn(loaders['train'], model, t=t)

        tr_res = utils.test(loaders['train'], base_model, criterion)
        te_res = utils.test(loaders['test'], base_model, criterion)
        tr_loss = tr_res['loss']
        tr_nll = tr_res['nll']
        tr_acc = tr_res['accuracy']
        tr_err = 100.0 - tr_acc
        te_loss = te_res['loss']
        te_nll = te_res['nll']
        te_acc = te_res['accuracy']
        te_err = 100.0 - te_acc
        values = [tr_loss, tr_nll, tr_err, te_nll, te_err]
        print(values)

        tr_res = utils.test(loaders['train'], model, criterion, t=t)
        te_res = utils.test(loaders['test'], model, criterion, t=t)
        tr_loss = tr_res['loss']
        tr_nll = tr_res['nll']
        tr_acc = tr_res['accuracy']
        tr_err = 100.0 - tr_acc
        te_loss = te_res['loss']
        te_nll = te_res['nll']
        te_acc = te_res['accuracy']
        te_err = 100.0 - te_acc
        values = [tr_loss, tr_nll, tr_err, te_nll, te_err]
        print(values)

        summary(base_model)

    else:
        model = architecture.base(num_classes=num_classes, **architecture.kwargs)
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['model_state'])

        loaders, num_classes = data.loaders(
            "CIFAR10",
            "CIFAR",
            128,
            2,
            "ResNet",
            False,
            shuffle_train=False
        )
        criterion = F.cross_entropy
        model.cuda()

        tr_res = utils.test(loaders['train'], model, criterion)
        te_res = utils.test(loaders['test'], model, criterion)
        tr_loss = tr_res['loss']
        tr_nll = tr_res['nll']
        tr_acc = tr_res['accuracy']
        tr_err = 100.0 - tr_acc
        te_loss = te_res['loss']
        te_nll = te_res['nll']
        te_acc = te_res['accuracy']
        te_err = 100.0 - te_acc
        values = [tr_loss, tr_nll, tr_err, te_nll, te_err]
        print(values)

        summary(model)

    if args.curve is not None and False:
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

        t = 0.5
        loaders, num_classes = data.loaders(
            "CIFAR10",
            "CIFAR",
            128,
            2,
            "ResNet",
            False,
            shuffle_train=False
        )
        criterion = F.cross_entropy
        utils.update_bn(loaders['train'], model, t=t)
        modelPoint = model.modelAt(t, architecture)
        modelPoint.cuda()

        summary(modelPoint)

        tr_res = utils.test(loaders['train'], modelPoint, criterion)
        te_res = utils.test(loaders['test'], modelPoint, criterion)
        tr_loss = tr_res['loss']
        tr_nll = tr_res['nll']
        tr_acc = tr_res['accuracy']
        tr_err = 100.0 - tr_acc
        te_loss = te_res['loss']
        te_nll = te_res['nll']
        te_acc = te_res['accuracy']
        te_err = 100.0 - te_acc
        values = [t, tr_loss, tr_nll, tr_err, te_nll, te_err]
        print(values)

if __name__ == '__main__':
    main()
