import os
import tabulate
import torch
import torch.nn.functional as F

import data
import models
import curves
import utils

import numpy as np
from pyhessian import hessian
from density_plot import get_esd_plot

# https://github.com/pytorch/pytorch/issues/49171#issuecomment-1056127453
from torch.autograd.functional import jacobian
from torch.nn.utils import stateless

from torchinfo import summary

import argparse

# python hessianCurve.py --dir=history --num_points=2 --dataset=CIFAR10 --data_path=CIFAR --ckpt=history/curve1-2/checkpoint-200.pt --curve=Bezier --model=PreResNet20 --transform=ResNet

def main():
    parser = argparse.ArgumentParser(description='DNN curve evaluation')
    parser.add_argument('--dir', type=str, default='/tmp/eval', metavar='DIR',
                        help='training directory (default: /tmp/eval)')

    parser.add_argument('--num_points', type=int, default=61, metavar='N',
                        help='number of points on the curve (default: 61)')

    parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                        help='dataset name (default: CIFAR10)')
    parser.add_argument('--use_test', action='store_true',
                        help='switches between validation and test set (default: validation)')
    parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                        help='transform name (default: VGG)')
    parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                        help='path to datasets location (default: None)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size (default: 128)')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='number of workers (default: 4)')

    parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                        help='model name (default: None)')
    parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                        help='curve type to use (default: None)')
    parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                        help='number of curve bends (default: 3)')

    parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                        help='checkpoint to eval (default: None)')

    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')

    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True

    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.transform,
        args.use_test,
        shuffle_train=True
    )

    inpFull, targetFull = next(iter(loaders['train']))

    architecture = getattr(models, args.model)
    curve = getattr(curves, args.curve)
    curve_model = curves.CurveNet(
        num_classes,
        curve,
        architecture.curve,
        args.num_bends,
        architecture_kwargs=architecture.kwargs,
    )
    curve_model.cuda()
    checkpoint = torch.load(args.ckpt)
    curve_model.load_state_dict(checkpoint['model_state'])

    criterion = F.cross_entropy
    regularizer = curves.l2_regularizer(args.wd)

    T = args.num_points
    ts = np.linspace(0.0, 1.0, T)
    tr_loss = np.zeros(T)
    tr_nll = np.zeros(T)
    tr_acc = np.zeros(T)
    te_loss = np.zeros(T)
    te_nll = np.zeros(T)
    te_acc = np.zeros(T)
    tr_err = np.zeros(T)
    te_err = np.zeros(T)
    dl = np.zeros(T)

    previous_weights = None

    columns = ['t', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']

    jacLossParamList = []
    jacOutParamList = []
    jacOutInpList = []
    hessTopEigenValList = []
    hessTopEigenVecList = []
    hessTraceList = []
    hessDensityEigenList = []
    hessDensityWeightList = []
    hessStraightList = []
    hessCurveList = []
    hessPerpList = []

    def toCPU(arr):
        if type(arr) is tuple or type(arr) is list:
            arr = list(arr)
            for j in range(len(arr)):
                arr[j] = toCPU(arr[j])
            return arr
        else:
            return arr.cpu()

    # makes the function params -> loss so that the jacobian and hessian functions compute with respect to parameters
    def loss(*params):
        out: torch.Tensor = stateless.functional_call(model, {n: p for n, p in zip(names, params)}, inp)
        return criterion(out, target)

    def outputs(*params):
        out: torch.Tensor = stateless.functional_call(model, {n: p for n, p in zip(names, params)}, inp)
        return out
    
    def get_xy(point, origin, vector_x, vector_y):
        return np.array([np.dot(point - origin, vector_x), np.dot(point - origin, vector_y)])

    # C:\ProgramData\Miniconda3\envs\torch\lib\site - packages\torch\autograd\__init__.py: 173: UserWarning: Using
    # backward()
    # with create_graph=True will create a reference cycle between the parameter and its gradient which can cause a memory leak.We recommend using autograd.grad when creating the graph to avoid this.If you have to use this function, make sure to reset the.grad fields of your parameters to None after use to break the cycle and avoid the leak.(Triggered internally at  C:\
    #     cb\pytorch_1000000000000\work\torch\csrc\autograd\engine.cpp: 1000.)
    # Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    def resetGrad(tmodel):
        for parameter in tmodel.parameters():
            parameter.grad = None

    w = list()
    curve_parameters = list(curve_model.net.parameters())
    # converting a tuple of tensors for each layer into a single vector
    for i in range(args.num_bends):
        w.append(np.concatenate([
            p.data.cpu().numpy().ravel() for p in curve_parameters[i::args.num_bends]
        ]))

    print('Weight space dimensionality: %d' % w[0].shape[0])

    # u is the straight line path between the 2 models on the curve
    # v is the vector perpendicular to u that the curved path detours in
    u = w[2] - w[0]
    dx = np.linalg.norm(u)
    u /= dx

    v = w[1] - w[0]
    v -= np.dot(u, v) * u
    dy = np.linalg.norm(v)
    v /= dy

    print("u v")
    print(u.shape)
    print(v.shape)


    modelAtT = architecture.base(num_classes=10, **architecture.kwargs)

    summary(modelAtT, (128, 3, 32, 32))

    offset = 0
    uList = []
    vList = []
    # reshapes u and v to match the layer structure of the model
    for parameter in modelAtT.parameters():
        size = np.prod(parameter.size())
        uList.append(torch.tensor(u[offset:offset + size].reshape(parameter.size())).cuda())
        vList.append(torch.tensor(v[offset:offset + size].reshape(parameter.size())).cuda())
        offset += size

    straightVector = tuple(uList)
    perpVector = tuple(vList)

    curveVectorList = []

    # bend_coordinates = np.stack(get_xy(p, w[0], u, v) for p in w)

    t = torch.FloatTensor([0.0]).cuda()
    for i, t_value in enumerate(ts):
        t.data.fill_(t_value)
        weights = curve_model.weights(t)
        if previous_weights is not None:
            dl[i] = np.sqrt(np.sum(np.square(weights - previous_weights)))

        previous_weights = weights.copy()

        # symetric difference tangent direction
        h = 0.001
        weightsForward = curve_model.weights(t + h)
        weightsBack = curve_model.weights(t - h)
        curveVector = weightsForward - weightsBack
        curveVector /= np.linalg.norm(curveVector)

        print(curveVector.shape)

        offset = 0
        cList = []
        for parameter in modelAtT.parameters():
            size = np.prod(parameter.size())
            cList.append(torch.tensor(curveVector[offset:offset + size].reshape(parameter.size())).cuda())
            offset += size

        curveVector = tuple(cList)

        # statistics on train loss and test loss
        utils.update_bn(loaders['train'], curve_model, t=t)
        tr_res = utils.test(loaders['train'], curve_model, criterion, regularizer, t=t)
        te_res = utils.test(loaders['test'], curve_model, criterion, regularizer, t=t)
        tr_loss[i] = tr_res['loss']
        tr_nll[i] = tr_res['nll']
        tr_acc[i] = tr_res['accuracy']
        tr_err[i] = 100.0 - tr_acc[i]
        te_loss[i] = te_res['loss']
        te_nll[i] = te_res['nll']
        te_acc[i] = te_res['accuracy']
        te_err[i] = 100.0 - te_acc[i]

        values = [t, tr_loss[i], tr_nll[i], tr_err[i], te_nll[i], te_err[i]]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
        if i % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

        # new calcs
        # creates a standalone model that stores the params from the curve model which computes them at runtime
        model = curve_model.modelAt(t, architecture)
        model.cuda()
        utils.update_bn(loaders['train'], model)
        model.eval()


        offset = 0
        batch_size = 128
        inp = inpFull[offset:offset + batch_size].cuda()
        target = targetFull[offset:offset + batch_size].cuda()
        print(inp.shape)

        names = list(n for n, _ in model.named_parameters())

        out = jacobian(loss, tuple(model.parameters()))
        out = list(out)
        for idx in range(len(out)):
            out[idx] = out[idx].cpu()
        jacLossParamList.append(out)

        resetGrad(model)

        # jacOutParamList.append(jacobian(outputs, tuple(model.parameters())).cpu())
        # resetGrad(model)

        offset = 0
        batch_size = 1
        inp = inpFull[offset:offset + batch_size].cuda()
        target = targetFull[offset:offset + batch_size].cuda()
        print(inp.shape)

        jacOutInpList.append(jacobian(model, inp).cpu())
        resetGrad(model)

        print(jacOutInpList[0].shape)

        # suspected windows error
        # if batch_size == 1:
        #     hessian_comp = hessian(model,
        #                            criterion,
        #                            data=next(iter(loaders['train'])),
        #                            cuda=True)
        # else:
        #     hessian_comp = hessian(model,
        #                            criterion,
        #                            dataloader=loaders['train'],
        #                            cuda=True)
        #
        # top_eigenvalues, top_eigenvectors = hessian_comp.eigenvalues()
        # trace = hessian_comp.trace()
        # density_eigen, density_weight = hessian_comp.density()
        #
        # resetGrad(model)
        #
        # hessTopEigenValList.append(top_eigenvalues)
        # hessTopEigenVecList.append(top_eigenvectors)
        # hessTraceList.append(trace)
        # hessDensityEigenList.append(density_eigen)
        # hessDensityWeightList.append(density_weight)
        #
        # print(top_eigenvalues)

        offset = 0
        batch_size = 128
        inp = inpFull[offset:offset + batch_size].cuda()
        target = targetFull[offset:offset + batch_size].cuda()
        print(inp.shape)

        # vector hessian products
        out = torch.autograd.functional.vhp(loss, inputs=tuple(model.parameters()), v=straightVector)
        out = toCPU(out)

        hessStraightList.append(out)
        resetGrad(model)

        out = torch.autograd.functional.vhp(loss, inputs=tuple(model.parameters()), v=perpVector)
        out = toCPU(out)

        hessPerpList.append(out)
        resetGrad(model)

        out = torch.autograd.functional.vhp(loss, inputs=tuple(model.parameters()), v=curveVector)
        out = toCPU(out)

        hessCurveList.append(out)

        resetGrad(model)

        curveVectorList.append(toCPU(curveVector))


    def stats(values, dl):
        min = np.min(values)
        max = np.max(values)
        avg = np.mean(values)
        int = np.sum(0.5 * (values[:-1] + values[1:]) * dl[1:]) / np.sum(dl[1:])
        return min, max, avg, int


    tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int = stats(tr_loss, dl)
    tr_nll_min, tr_nll_max, tr_nll_avg, tr_nll_int = stats(tr_nll, dl)
    tr_err_min, tr_err_max, tr_err_avg, tr_err_int = stats(tr_err, dl)

    te_loss_min, te_loss_max, te_loss_avg, te_loss_int = stats(te_loss, dl)
    te_nll_min, te_nll_max, te_nll_avg, te_nll_int = stats(te_nll, dl)
    te_err_min, te_err_max, te_err_avg, te_err_int = stats(te_err, dl)

    print('Length: %.2f' % np.sum(dl))
    print(tabulate.tabulate([
            ['train loss', tr_loss[0], tr_loss[-1], tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int],
            ['train error (%)', tr_err[0], tr_err[-1], tr_err_min, tr_err_max, tr_err_avg, tr_err_int],
            ['test nll', te_nll[0], te_nll[-1], te_nll_min, te_nll_max, te_nll_avg, te_nll_int],
            ['test error (%)', te_err[0], te_err[-1], te_err_min, te_err_max, te_err_avg, te_err_int],
        ], [
            '', 'start', 'end', 'min', 'max', 'avg', 'int'
        ], tablefmt='simple', floatfmt='10.4f'))



    # for i in range(len(jac)):
    #     print(torch.norm(jac[i]))

    np.savez(
        os.path.join(args.dir, 'curveHessian.npz'),
        ts=ts,
        dl=dl,
        tr_loss=tr_loss,
        tr_loss_min=tr_loss_min,
        tr_loss_max=tr_loss_max,
        tr_loss_avg=tr_loss_avg,
        tr_loss_int=tr_loss_int,
        tr_nll=tr_nll,
        tr_nll_min=tr_nll_min,
        tr_nll_max=tr_nll_max,
        tr_nll_avg=tr_nll_avg,
        tr_nll_int=tr_nll_int,
        tr_acc=tr_acc,
        tr_err=tr_err,
        tr_err_min=tr_err_min,
        tr_err_max=tr_err_max,
        tr_err_avg=tr_err_avg,
        tr_err_int=tr_err_int,
        te_loss=te_loss,
        te_loss_min=te_loss_min,
        te_loss_max=te_loss_max,
        te_loss_avg=te_loss_avg,
        te_loss_int=te_loss_int,
        te_nll=te_nll,
        te_nll_min=te_nll_min,
        te_nll_max=te_nll_max,
        te_nll_avg=te_nll_avg,
        te_nll_int=te_nll_int,
        te_acc=te_acc,
        te_err=te_err,
        te_err_min=te_err_min,
        te_err_max=te_err_max,
        te_err_avg=te_err_avg,
        te_err_int=te_err_int,

        jacLossParamList=np.array(jacLossParamList, dtype=object),
        # jacOutParamList=np.array(jacOutParamList, dtype=object),
        jacOutInpList=np.array(jacOutInpList, dtype=object),
        # hessTopEigenValList=np.array(hessTopEigenValList, dtype=object),
        # hessTopEigenVecList=np.array(hessTopEigenVecList, dtype=object),
        # hessTraceList=np.array(hessTraceList, dtype=object),
        # hessDensityEigenList=np.array(hessDensityEigenList, dtype=object),
        # hessDensityWeightList=np.array(hessDensityWeightList, dtype=object),
        hessStraightList=np.array(hessStraightList, dtype=object),
        hessCurveList=np.array(hessCurveList, dtype=object),
        hessPerpList=np.array(hessPerpList, dtype=object),
        curveVectorList=np.array(toCPU(curveVectorList), dtype=object),
        straightVector=np.array(toCPU(straightVector), dtype=object),
        perpVector=np.array(toCPU(perpVector), dtype=object),
    )

if __name__ == '__main__':
    main()