'''Train CIFAR10 with PyTorch.'''
import torch.backends.cudnn as cudnn
import argparse

from tqdm import tqdm
import torch
import os
import torchvision
import numpy as np
from torch.utils import data
import torchvision.transforms.functional as tf
# from CSV_ASSPP_N_Unet_CAT import ASSPP_N_Unet_CLS
from my_dataset.my_datasets import  MyValDataSet_cls
from sklearn import metrics
import csv
# from model import resnet101
import random
from torchvision import models


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--imgs_train_path', '-it', type=str,
                    default='//mnt/ai2020/orton/dataset/hua_lu_fu_sai/test20210811/test_image/',
                    help='imgs train data path.')
parser.add_argument('--resize', default=224, type=int, help='resize shape')
parser.add_argument('--batch_size', default=1, type=int, help='batchsize')
parser.add_argument('--device', default='cuda', type=str, help='use cuda')
parser.add_argument('--start_epoch', '-s', default=0, type=int, help='start epoch')
parser.add_argument('--end_epoch', '-e', default=300, type=int, help='end epoch')
parser.add_argument('--devicenum', default='0', type=str, help='use devicenum')
args = parser.parse_args()

RANDOM_SEED = 6666
batch_size = 16
INPUT_CHANNEL = 4
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
random.seed(RANDOM_SEED)


result_path = '/mnt/ai2019/zxg_FZU/dataset/result/my_cls_result/'
if not os.path.isdir(result_path):
    os.mkdir(result_path)

# os.environ['CUDA_VISIBLE_DEVICES'] = args.devicenum
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
torch.backends.cudnn.enabled = True
# device = args.device  # 是否使用cuda
# Data
print('==> Preparing data..')
############# Load testing data
data_train_root = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/test/images/'
data_train_root_mask = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/CCE-NET/test/coarse_mask3/'
data_train_list = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/my_test_label_cls.txt'
testloader = data.DataLoader(MyValDataSet_cls(data_train_root, data_train_root_mask, data_train_list), batch_size=1,
                             shuffle=False,
                             num_workers=8,
                             pin_memory=True)

# Model
print('==> Building model..')
model_name = 'mobilenet_v2'
print(model_name)
# net = mobilenet_v2(num_classes=4)
# net = ASPP_N_Unet_CLS(3, 2)
# net = resnet101()
# net = n_ClsNet.M_Net_CAC_Slide_CLS(3, 5)
# net = ARL.arlnet50(pretrained=True)
net = models.resnet50(num_classes=5)
# num_ftrs = net.fc.in_features
#
# net.fc = nn.Linear(num_ftrs, 5)

# net.load_state_dict(torch.load('/mnt/ai2019/zxg_FZU/classification_projects/ARL/checkpoint/my_cls_best_acc.pth', map_location=device), strict=False)
net.load_state_dict(torch.load('/mnt/ai2019/zxg_FZU/my_first_paper_source_code/checkpoint/my_cls_checkpoint/ResNet50.pth', map_location=device), strict=False)
# num_ftrs = net.fc.in_features
#
# net.fc = nn.Linear(num_ftrs, 5)
net = net.to(device)
net.eval()


def cla_evaluate(label, binary_score, pro_score):
    acc = metrics.accuracy_score(label, binary_score)
    AP = metrics.average_precision_score(label, pro_score)
    auc = metrics.roc_auc_score(label, pro_score)
    f1_score = metrics.f1_score(label, binary_score, average='macro')
    precision = metrics.precision_score(label, binary_score)
    recall = metrics.recall_score(label, binary_score, average='macro')
    jaccard = metrics.jaccard_score(label, binary_score, average='macro')
    CM = metrics.confusion_matrix(label, binary_score)
    sens = float(CM[1, 1]) / float(CM[1, 1] + CM[1, 0])
    spec = float(CM[0, 0]) / float(CM[0, 0] + CM[0, 1])
    return acc, AP, auc, f1_score, precision, recall, jaccard, sens, spec

rot = torchvision.transforms.functional.rotate
def sum_8(model, tensor,coarsemask):
    batch = torch.cat([tensor, tensor.flip([-1]), tensor.flip([-2]), tensor.flip([-1,-2]), rot(tensor,90).flip([-1]), rot(tensor,90).flip([-2]), rot(tensor,90).flip([-1,-2]), rot(tensor,90)], 0)

    # pred = model(batch).detach()

    if coarsemask == None:
        pred = model(batch).detach()
    else:
        batch_coarse = torch.cat([coarsemask, coarsemask.flip([-1]), coarsemask.flip([-2]), coarsemask.flip([-1, -2]),
                                  rot(coarsemask, 90).flip([-1]), rot(coarsemask, 90).flip([-2]),
                                  rot(coarsemask, 90).flip([-1, -2]), rot(coarsemask, 90)], 0)
        pred = model(batch, batch_coarse)

        ## M_Net
        # pred, side_5, side_6, side_7, side_8 = model(batch, batch_coarse)
        # pred = torch.log(softmax_2d(side_8) + EPS)
    return pred[:1]+pred[1:2]+pred[2:3]+pred[3:4]+pred[4:5]+pred[5:6]+pred[6:7]+pred[7:]



rot = tf.rotate
def sum_5(model, image,coarsemask):
    rot_90 = torch.rot90(image, 1, [2, 3])
    rot_180 = torch.rot90(image, 2, [2, 3])
    rot_270 = torch.rot90(image, 3, [2, 3])
    hor_flip = torch.flip(image, [-1])
    ver_flip = torch.flip(image, [-2])
    image = torch.cat([image, rot_90, rot_180, rot_270, hor_flip, ver_flip], dim=0)
    if coarsemask==None:
        pred = model(image)
    else:
        rot_90_cm = torch.rot90(coarsemask, 1, [2, 3])
        rot_180_cm = torch.rot90(coarsemask, 2, [2, 3])
        rot_270_cm = torch.rot90(coarsemask, 3, [2, 3])
        hor_flip_cm = torch.flip(coarsemask, [-1])
        ver_flip_cm = torch.flip(coarsemask, [-2])
        coarsemask = torch.cat([coarsemask, rot_90_cm, rot_180_cm, rot_270_cm, hor_flip_cm, ver_flip_cm], dim=0)

        with torch.no_grad():
            inputs = torch.cat((image, coarsemask), dim=1)
        pred = model(inputs)

        ### M-Net_CAC
        # pred, side_5, side_6, side_7, side_8 = net(image, coarsemask)
        # pred = torch.log(softmax_2d(side_8) + EPS)

    # with torch.no_grad():
    #     image = torch.cat((image, coarsemask), dim=1)
    pred = pred[0:1] + pred[1:2]+ pred[2:3] + pred[3:4]+ pred[4:5] + pred[5:6]
    return pred

def train_val():
    test_for_csv = []
    pre_score_all = []
    prob_val_all = []
    label_val_all = []
    pred_prob = []
    correct_val = 0
    total_val = 0
    pred_score = []
    label_val = []
    metrics_for_csv = []
    with torch.no_grad():
        with tqdm(total=len(testloader), ncols=70, ascii=True) as t:
            for batch_idx, (inputs, coarsemask, label, name) in enumerate(testloader):
                t.set_description("Val(Epoch {}/{})".format(1, 1))

                image, label, coarsemask = inputs.to(device), label.to(device), coarsemask.to(device)
                coarsemask = coarsemask.unsqueeze(1).cuda()
                net.eval()

                # out = sum_8(net,image,coarsemask)
                # with torch.no_grad():
                #     inputs = torch.cat((image, coarsemask), dim=1)
                out = net(inputs)
                correct_val += (torch.max(out, 1)[1].view(label.size()).data == label.data).sum()
                total_val += testloader.batch_size
                epoch_val_acc = ((correct_val.item()) / total_val)

                pred_label = torch.max(out, dim=1)[1].cpu().data.numpy().item()

                pred_score.append(torch.softmax(out[0], dim=0).cpu().data.numpy())
                label_val.append(label[0].cpu().data.numpy())

                img_name = "".join(name)
                test_for_csv.append([img_name, pred_label])
                t.set_postfix(batch_idx='{:.3f}'.format(batch_idx), )
                t.update(1)


            pro_score = np.array(pred_score)
            label_val = np.array(label_val)
            num = len(pro_score[1])

            pro_score_all = np.array(pro_score)
            binary_score_all = np.eye(num)[np.argmax(np.array(pro_score), axis=-1)]
            label_val_all = np.eye(num)[np.int64(np.array(label_val))]
            if num == 3:
                metrics_for_csv.append(['melanoma', 'seborrheic_keratosis','normal'])
            else:
                metrics_for_csv.append(['baso', 'eosi','lymp','mono','mono'])

            metrics_for_csv.append(['acc', 'AP', 'auc', 'f1_score', 'precision', 'recall','jaccard', 'sens', 'spec'])
            for i in range(num):
                label_val_cls0 = label_val_all[:, i-1]
                pred_prob_cls0 = pro_score_all[:, i-1]
                binary_score_cls0 = binary_score_all[:, i-1]
                acc, AP, auc, f1_score, precision, recall, jaccard, sens, spec = cla_evaluate(label_val_cls0,
                                                                                              binary_score_cls0,
                                                                                              pred_prob_cls0)

                line_test_cls0 = "test:acc=%f,AP=%f,auc=%f,f1_score=%f,precision=%f,recall=%f,sens=%f,spec=%f\n" % (
                    acc, AP, auc, f1_score, precision, recall, sens, spec)
                print(line_test_cls0)
                metrics_for_csv.append([acc, AP, auc, f1_score, precision, recall, jaccard, sens, spec])


            # results_file = open(result_path + '/wbc_M_Net_CAC_Slide_CLS_Input_Cat_img_label.csv', 'w', newline='')
            # csv_writer = csv.writer(results_file, dialect='excel')
            # for row in test_for_csv:
            #     csv_writer.writerow(row)

            results_file = open(result_path + '/ResNet50_metrics.csv', 'w', newline='')
            csv_writer = csv.writer(results_file, dialect='excel')
            for row in metrics_for_csv:
                csv_writer.writerow(row)


if __name__ == '__main__':
    train_val()
