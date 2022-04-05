import pandas as pd
import tqdm
import torch
import os
import pandas

from utils.metrics import ConfusionMatrixAsan, ConfusionMatrixBrats
from config.config import ASAN_DATA_PATH, BRATS2015_DATA_PATH, BRATS2020_DATA_PATH


def evaluate_test(segmentation_net, data_test, num_classes, device, obfuscator_net=None, dataset=None, logger=None,
                  result_path=None):
    segmentation_net.eval()
    if obfuscator_net:
        obfuscator_net.eval()

    ASAN_TEST_DATA_PATH = os.path.join(ASAN_DATA_PATH, 'data_preprocessed/CHD_3D_Group8', 'test')
    if dataset == 'asan':
        idx_li = []
        for i in range(81, 121):
            prefix = str(i)
            if len(prefix) < 3:
                prefix = '0' + prefix
            file_name = prefix + '_dia_00.pkl'
            idx_li.append(data_test.file_paths.index(os.path.join(ASAN_TEST_DATA_PATH, file_name)))
            file_name = prefix + '_sys_00.pkl'
            idx_li.append(data_test.file_paths.index(os.path.join(ASAN_TEST_DATA_PATH, file_name)))
    elif 'brats' in dataset:
        pass
        # ToDo

    df = pd.DataFrame(columns=['mIoU (class 0)', 'mIoU (class 1)', 'mIoU (class 2)', 'mIoU (class 3)',
                               'mIoU (all classes)', 'mDice (class 0)', 'mDice (class 1)', 'mDice (class 2)',
                               'mDice (class 3)', 'mDice (all classes)'])
    with torch.no_grad():
        for i in range(len(idx_li[:])):
            start_point = idx_li[i]
            if i == len(idx_li) - 1:
                end_point = len(data_test)
            else:
                end_point = idx_li[i + 1]

            stack_li_x = []
            stack_li_y = []
            for j in range(start_point, end_point):
                instance_test = data_test[j]
                stack_li_x.append(instance_test[0])
                stack_li_y.append(instance_test[1])

            x = torch.stack(stack_li_x)
            y = torch.stack(stack_li_y)

            x = x.to(device)
            y = y.to(device)

            if obfuscator_net:
                delta = obfuscator_net(x)
                x_o = x + delta
                x_pred = segmentation_net(x_o)
            else:
                x_pred = segmentation_net(x)

            if dataset == 'asan':
                confmat = ConfusionMatrixAsan(num_classes)
            elif 'brats' in dataset:
                confmat = ConfusionMatrixBrats(num_classes)
                # ToDo
            confmat.update(y.flatten(), x_pred.argmax(1).flatten())
            _, _, mIoU, mDice = confmat.compute()
            df = df.append({'mIoU (class 0)': round(mIoU[0].item() * 100, 2),
                            'mIoU (class 1)': round(mIoU[1].item() * 100, 2),
                            'mIoU (class 2)': round(mIoU[2].item() * 100, 2),
                            'mIoU (class 3)': round(mIoU[3].item() * 100, 2),
                            'mIoU (all classes)': round(mIoU.mean().item() * 100, 2),
                            'mDice (class 0)': round(mDice[0].item() * 100, 2),
                            'mDice (class 1)': round(mDice[1].item() * 100, 2),
                            'mDice (class 2)': round(mDice[2].item() * 100, 2),
                            'mDice (class 3)': round(mDice[3].item() * 100, 2),
                            'mDice (all classes)': round(mDice.mean().item() * 100, 2)},
                           ignore_index=True)
            logger.info("== Test == " + confmat.__str__())
    df.to_csv(os.path.join(result_path, 'results.csv'))

    return


def evaluate(segmentation_net, dataloader, num_classes, device, obfuscator_net=None, dataset=None):
    segmentation_net.eval()
    if obfuscator_net:
        obfuscator_net.eval()

    if dataset == 'asan':
        confmat = ConfusionMatrixAsan(num_classes)
    elif 'brats' in dataset:
        confmat = ConfusionMatrixBrats(num_classes)

    with torch.no_grad():
        for x, y in tqdm.tqdm(dataloader):

            x = x.to(device)
            y = y.to(device)

            if obfuscator_net:
                delta = obfuscator_net(x)
                x_o = x + delta
                x_pred = segmentation_net(x_o)
            else:
                x_pred = segmentation_net(x)

            confmat.update(y.flatten(), x_pred.argmax(1).flatten())
            confmat.reduce_from_all_processes()

    return confmat
