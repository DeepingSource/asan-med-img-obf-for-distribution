import os
import sys
import tqdm
import shutil
import random
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from utils.loss import DiceLoss
from utils.utils import one_hot
from utils.validation import evaluate
from utils.file import get_result_path
from model.unet3d import UNet3D, ResidualUNet3D
from utils.logger import get_logger, AverageMeter
from utils.data import get_data_specs, get_dataset
from utils.metrics import ConfusionMatrixAsan, ConfusionMatrixBrats, DiceCoefficient


def parse_arguments():
    parser = argparse.ArgumentParser(description="Trains a neural network")
    # Standard parameters
    parser.add_argument("--dataset", required=True, help="Training dataset")
    parser.add_argument("--arch", required=True, help="Model architecture")
    parser.add_argument("--seed", type=int, default=1337, help="Seed used")
    # Data parameters
    parser.add_argument("--grouping", type=int, default=128)
    # Optimization options
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
    parser.add_argument("--learning-rate", type=float, required=True, help="Learning Rate")
    # Folder structure
    parser.add_argument("--subfolder", type=str, default="", help="Subfolder to store results in")
    parser.add_argument("--postfix", type=str, default="", help="Attach postfix to model name")
    # MISC
    parser.add_argument("--workers", type=int, default=6, help="Number of data loading workers")
    parser.add_argument("--print-freq", default=200, type=int, help="print frequency")
    # Existing model loading
    parser.add_argument("--weights", type=str, default="", help="Model path to load")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    # Setting the seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Result path
    subfolder = os.path.join("segmentation", args.subfolder)
    result_path = get_result_path(dataset_name=args.dataset,
                                  arch=args.arch,
                                  seed=args.seed,
                                  subfolder=subfolder,
                                  postfix=args.postfix)

    # Logging
    logger = get_logger(result_path)

    logger.info("Python version : {}".format(sys.version.replace('\n', ' ')))
    logger.info("Torch version : {}".format(torch.__version__))
    logger.info("Cudnn version : {}".format(torch.backends.cudnn.version()))
    logger.info("Model Path : {}".format(result_path))

    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        logger.info("{} : {}".format(key, value))

    # Save this file
    file_save_path = os.path.join(result_path, 'code')
    if not os.path.isdir(file_save_path):
        os.makedirs(file_save_path)
    shutil.copy(sys.argv[0], os.path.join(file_save_path, sys.argv[0]))

    # Load Data
    num_classes, in_channels, image_size, train_transform, test_transform = get_data_specs(dataset_name=args.dataset)
    data_train, data_test = get_dataset(dataset_name=args.dataset,
                                        train_transform=train_transform,
                                        test_transform=test_transform,
                                        grouping=args.grouping)

    # DataLoaders
    dataloader_train = torch.utils.data.DataLoader(data_train,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers)
    dataloader_test = torch.utils.data.DataLoader(data_test,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.workers)

    # Load model
    if args.arch == 'unet3d':
        segmentation_net = ResidualUNet3D(in_channels=in_channels, out_channels=num_classes, num_levels=4, f_maps=12,
                                          num_groups=1)
    else:
        raise ValueError
    # TODO: Data parallel
    segmentation_net.to(device)
    if args.weights != "":
        checkpoint = torch.load(args.weights)
        segmentation_net.load_state_dict(checkpoint['segmentation'], strict=True)
    # segmentation_net = torch.nn.DataParallel(segmentation_net).cuda()

    # Optimizer
    xent = nn.CrossEntropyLoss()
    dice_weights = torch.ones(num_classes).to(device)
    dice_weights[0] = 0.01
    # dice_weights[0] = 1.00
    dice = DiceLoss(weight=dice_weights, normalization='sigmoid')
    optimizerF = torch.optim.Adam(segmentation_net.parameters(), lr=args.learning_rate)

    # AverageMeters
    loss_xent_avgm = AverageMeter()
    loss_dice_avgm = AverageMeter()
    loss_avgm = AverageMeter()

    best_dice = 0
    for epoch in range(args.epochs):
        segmentation_net.train()

        loss_xent_avgm.reset()
        loss_dice_avgm.reset()
        loss_avgm.reset()

        # if args.dataset == 'asan':
        #     confmat_train = ConfusionMatrixAsan(num_classes)
        # elif 'brats' in args.dataset:
        #     confmat_train = ConfusionMatrixBrats(num_classes)
        # confmat_train.reset()
        
        for batch_idx, (x, y) in tqdm.tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            x = x.to(device)
            y = y.to(device)
            # temp = y.shape[0] * y.shape[1] * y.shape[2] * y.shape[3]
            # print("temp", temp)

            # Feed into net
            x_pred = segmentation_net(x)

            # Flatten out x_pred and y
            x_pred_flat = x_pred.transpose(0, 1).transpose(0, 4).reshape(-1, num_classes)
            y_flat = y.flatten().to(torch.long)
            y_one_hot = one_hot(y, num_classes)

            # Softmax
            x_pred_flat_smax = torch.nn.functional.softmax(x_pred_flat, dim=-1)

            # Cross entropy loss
            loss_xent = xent(x_pred_flat_smax, y_flat)
            loss_dice = dice(x_pred, y_one_hot)
            loss = loss_dice

            loss_xent_avgm.update(loss_xent.item(), x.size(0))
            loss_dice_avgm.update(loss_dice.item(), x.size(0))
            loss_avgm.update(loss.item(), x.size(0))

            # Optimization
            optimizerF.zero_grad()
            loss.backward()
            optimizerF.step()

            # # Confusion Matrix Update
            # confmat_train.update(y.flatten(), x_pred.argmax(1).flatten())
            # confmat_train.reduce_from_all_processes()

            if (batch_idx+1) % args.print_freq == 0:
                logger.info('Epoch: {} LR: {lr:5f}\t'
                            'Xent {loss_xent.val:.4f} ({loss_xent.avg:.4f})\t'
                            'Dice {loss_dice.val:.4f} ({loss_dice.avg:.4f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch,
                    lr=optimizerF.param_groups[0]["lr"],
                    loss_xent=loss_xent_avgm,
                    loss_dice=loss_dice_avgm,
                    loss=loss_avgm))


        # # Train Confusion Matrix
        # logger.info("== Train == Epoch: {}\t".format(epoch) + confmat_train.__str__())

        # Test
        confmat_test = evaluate(segmentation_net=segmentation_net,
                                dataloader=dataloader_test,
                                num_classes=num_classes,
                                device=device,
                                obfuscator_net=None,
                                dataset=args.dataset)
        logger.info("== Test == Epoch: {}\t".format(epoch) + confmat_test.__str__())

        # Store checkpoint
        save_path = os.path.join(result_path, 'checkpoint.pth')
        torch.save(
            {
                'segmentation': segmentation_net.state_dict(),
                'optimizerF': optimizerF.state_dict(),
            }, save_path)

        if args.dataset == 'asan':
            _, _, _, cur_dice = confmat_test.compute()
            cur_dice = cur_dice.mean().item()
        elif 'brats' in args.dataset:
            # cur_dice is complete dice score in this case
            _, _, _, _, cur_dice, _, _ = confmat_test.compute()

        if cur_dice > best_dice:
            save_path = os.path.join(result_path, 'best_checkpoint.pth')
            torch.save(
                {
                    'segmentation': segmentation_net.state_dict(),
                    'optimizerF': optimizerF.state_dict(),
                }, save_path)
            best_dice = cur_dice
            print("New best dice:", best_dice)


if __name__ == "__main__":
    main()
