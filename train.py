import argparse
import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F

import curves
import data
import models
import utils

from torchinfo import summary


def learning_rate_schedule(base_lr, epoch, total_epochs):
    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr


def parseArgs():
    parser = argparse.ArgumentParser(description='DNN curve training')

    # use defaults
    parser.add_argument('--dir', type=str, default='history', metavar='DIR',
                        help='training directory (default: history)')
    parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                        help='dataset name (default: CIFAR10)')
    parser.add_argument('--use_test', action='store_true',
                        help='switches between validation and test set (default: validation)')
    parser.add_argument('--data_path', type=str, default='CIFAR', metavar='PATH',
                        help='path to datasets location (default: CIFAR)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size (default: 128)')
    #########################################

    # turn off data augmentation with ResNetNoAugment
    parser.add_argument('--transform', type=str, default='ResNet', metavar='TRANSFORM',
                        help='transform name (default: ResNet)')

    # use 2 in google colab
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of workers (default: 4)')

    # PreResNet20 or PreResNet20_4x for constant activation space size
    parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                        help='model name (default: None)')

    # leave default for not using curve
    parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                        help='curve type to use (default: None)')
    parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                        help='number of curve bends (default: 3)')
    parser.add_argument('--init_start', type=str, default=None, metavar='CKPT',
                        help='checkpoint to init start point (default: None)')
    parser.add_argument('--fix_start', dest='fix_start', action='store_true',
                        help='fix start point (default: off)')
    parser.add_argument('--init_end', type=str, default=None, metavar='CKPT',
                        help='checkpoint to init end point (default: None)')
    parser.add_argument('--fix_end', dest='fix_end', action='store_true',
                        help='fix end point (default: off)')
    parser.set_defaults(init_linear=True)
    parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
                        help='turns off linear initialization of intermediate points (default: on)')
    #############################################

    parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                        help='checkpoint to resume training from (default: None)')

    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--save_freq', type=int, default=50, metavar='N',
                        help='save frequency (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')

    # train on subsets of dataset
    parser.add_argument('--parts', type=int, default=1, metavar='N',
                        help='number of dataset partitions (default: 1)')
    parser.add_argument('--offset', type=int, default=0, metavar='N',
                        help='which partition to use (default: 0)')

    parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: None)')

    args = parser.parse_args()

    return args


# https://pytorch.org/docs/stable/notes/windows.html#multiprocessing-error-without-if-clause-protection
def main(args=None):

    if args is None:
        args = parseArgs()

    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    torch.backends.cudnn.benchmark = True

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.transform,
        args.use_test,
        offset=args.offset,
        partitions=args.parts
    )

    utils.dataDist(loaders["train"], num_classes)

    architecture = getattr(models, args.model)

    if args.curve is None:
        model = architecture.base(num_classes=num_classes, **architecture.kwargs)
        summary(model, input_size=(128, 3, 32, 32))
    else:
        curve = getattr(curves, args.curve)
        model = curves.CurveNet(
            num_classes,
            curve,
            architecture.curve,
            args.num_bends,
            args.fix_start,
            args.fix_end,
            architecture_kwargs=architecture.kwargs,
        )
        base_model = None
        if args.resume is None:
            for path, k in [(args.init_start, 0), (args.init_end, args.num_bends - 1)]:
                if path is not None:
                    if base_model is None:
                        base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
                    checkpoint = torch.load(path)
                    print('Loading %s as point #%d' % (path, k))
                    base_model.load_state_dict(checkpoint['model_state'])
                    model.import_base_parameters(base_model, k)
            if args.init_linear:
                print('Linear initialization.')
                model.init_linear()
    model.cuda()

    criterion = F.cross_entropy
    regularizer = None if args.curve is None or args.wd == 0 else curves.l2_regularizer(args.wd)
    optimizer = torch.optim.SGD(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd if args.curve is None else 0.0
    )

    start_epoch = 1
    if args.resume is not None:
        print('Resume training from %s' % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'time']

    utils.save_checkpoint(
        args.dir,
        start_epoch - 1,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict()
    )

    has_bn = utils.check_bn(model)
    test_res = {'loss': None, 'accuracy': None, 'nll': None}
    for epoch in range(start_epoch, args.epochs + 1):
        time_ep = time.time()

        lr = learning_rate_schedule(args.lr, epoch, args.epochs)
        utils.adjust_learning_rate(optimizer, lr)

        train_res = utils.train(loaders['train'], model, optimizer, criterion, regularizer)
        if args.curve is None or not has_bn:
            test_res = utils.test(loaders['test'], model, criterion, regularizer)

        if epoch % args.save_freq == 0:
            utils.save_checkpoint(
                args.dir,
                epoch,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict()
            )

        time_ep = time.time() - time_ep
        values = [epoch, lr, train_res['loss'], train_res['accuracy'], test_res['nll'],
                  test_res['accuracy'], time_ep]

        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
        if epoch % 40 == 1 or epoch == start_epoch:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

    if args.epochs % args.save_freq != 0:
        utils.save_checkpoint(
            args.dir,
            args.epochs,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )


if __name__ == '__main__':
    a = parseArgs()
    print(a)
    main(args=a)