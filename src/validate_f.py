import os
import sys
import tqdm
import shutil
import random
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from utils.validation import evaluate_test, evaluate
from utils.file import get_result_path
from model.unet3d import UNet3D, ResidualUNet3D
from utils.logger import get_logger
from utils.data import get_data_specs, get_testset


def parse_arguments():
    parser = argparse.ArgumentParser(description="Trains a neural network")
    # Standard parameters
    parser.add_argument("--seed", type=int, default=1337, help="Seed used")
    parser.add_argument("--dataset", required=True, help="Training dataset")
    parser.add_argument("--arch", required=True, help="Model architecture")
    # Data parameters
    parser.add_argument("--grouping", type=int, default=128)
    # Optimization options
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
    # Folder structure
    parser.add_argument("--subfolder", type=str, default="", help="Subfolder to store results in")
    parser.add_argument("--postfix", type=str, default="", help="Attach postfix to model name")
    # MISC
    parser.add_argument("--workers", type=int, default=6, help="Number of data loading workers")
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
    subfolder = os.path.join("segmentation_test", args.subfolder)
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
    data_test = get_testset(dataset_name=args.dataset,
                            test_transform=test_transform,
                            grouping=args.grouping)

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

    # Performance for each test instance
    evaluate_test(segmentation_net=segmentation_net,
                                 data_test=data_test,
                                 num_classes=num_classes,
                                 device=device,
                                 obfuscator_net=None,
                                 dataset=args.dataset,
                                 logger=logger,
                                 result_path=result_path)

    # Average test performance
    dataloader_test = torch.utils.data.DataLoader(data_test,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.workers)
    confmat_test = evaluate(segmentation_net=segmentation_net,
                            dataloader=dataloader_test,
                            num_classes=num_classes,
                            device=device,
                            obfuscator_net=None,
                            dataset=args.dataset)
    logger.info("== Average Test == " + confmat_test.__str__())


if __name__ == "__main__":
    main()
