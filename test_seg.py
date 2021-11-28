import os
from models_00 import *
import torch
from models_00.metrics import Miou
import cv2
# from my_skin_lesion_datasets_with_cam import MyDataSet_seg, MyValDataSet_seg, MyTestDataSet_seg
from my_dataset.my_datasets_ori import MyTestDataSet_seg

import argparse
from torch.utils import data
import models_00.other_models.cenet as cenet


parser = argparse.ArgumentParser(description='my semi-supervised learning')
parser.add_argument('--imgs_train_path', '-it', type=str, default='/media/user/DA18EBFA09C1B27D/segmentation/train/images', help='imgs train data path.')
parser.add_argument('--labels_train_path', '-lt', type=str, default='/media/user/DA18EBFA09C1B27D/segmentation/train/masks', help='labels train data path.')
parser.add_argument('--imgs_val_path', '-iv', type=str, default='/mnt/nfs/zxg/dataset/skin_lesion_data/test/Images', help='imgs val data path.')
parser.add_argument('--labels_val_path', '-lv', type=str, default='/mnt/nfs/zxg/dataset/skin_lesion_data/test/Annotation', help='labels val data path.')
parser.add_argument('--resize', default=512, type=int, help='resize shape')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--start_epoch', '-s', default=0, type=int, help='start epoch')
parser.add_argument('--end_epoch', '-e', default=100, type=int, help='end epoch')
parser.add_argument('--times', '-t', default=1, type=int, help='val')
parser.add_argument('--device', default='cpu', type=str, help='use cuda')
parser.add_argument('--tb_path', type=str, default='log/Unet', help='tensorboard path')
parser.add_argument('--checkpoint', type=str, default='checkpoint/Unet/', help='checkpoint path')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# result_path='/mnt/ai2019/zxg_FZU/dataset/result/my_wbc_result/draw_ROC/U_Net/'
# result_path='/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/MB_DCNN/M_Net_CAC/'
# result_path='/mnt/ai2019/zxg_FZU/dataset/result/skin_lesion/draw_roc/M_Unet/'
result_roc_path='/mnt/ai2019/zxg_FZU/dataset/result/my_wbc_result/old_draw_roc/cenet_roc/'
# masks_224_path='/mnt/ai2019/zxg_FZU/dataset/result/my_wbc_result/masks_224/'
# img_224_path='/mnt/ai2019/zxg_FZU/dataset/result/my_wbc_result/images_224/'

args = parser.parse_args()
if not os.path.isdir(result_roc_path):
    os.mkdir(result_roc_path)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

print('==> Preparing data..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPS = 1e-12

RANDOM_SEED = 6666
batch_size = 16
INPUT_CHANNEL = 4
# Model
print('==> Building model..')
# net = AGNET.AG_Net(3,2)
# net =psp.DeepLabV3(encoder_name='resnet34',encoder_weights=None,classes=2).to(device)
# net = AGNET.M_Net_CAC(6, 2)
# net = deeplabv3plus(num_classes=2)
# net =psp.PSPNet(encoder_name='resnet34',encoder_weights=None,classes=2).to(device)
net = cenet.CE_Net_(3, 2)
# net = attunet.AttU_Net(3, 2)

net.load_state_dict(torch.load('/mnt/ai2019/zxg_FZU/my_first_paper_source_code/checkpoint/my_wbc_checkpoint/wbc_cenet.pth')['net'], strict=True)

net = net.to(device)
net.eval()
softmax_2d = nn.Softmax2d()
criterion = nn.NLLLoss2d()

############# Load testing data
data_test_root = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/test/'
data_test_root_mask = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/CCE-NET/test/coarse_mask3/'
data_test_list = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/test_img_label_seg.txt'
testloader = data.DataLoader(MyTestDataSet_seg(data_test_root, data_test_list, root_path_coarsemask=data_test_root_mask), batch_size=1, shuffle=False,
                             num_workers=8,
                             pin_memory=True)

def test_data():
    with torch.no_grad():
        test_miou = 0
        test_mdice = 0
        test_pre = 0
        test_recall = 0
        test_F1score = 0
        # dice=0

        for batch_idx, (image, coarsemask, label, img_path) in enumerate(testloader):
            batch_idx += 1
            image, label, coarsemask = image.to(device), label.to(device), coarsemask.to(device)
            coarsemask = coarsemask.unsqueeze(1).cuda()
            targets = label.long()
            img = image

            ##
            rot_90 = torch.rot90(image, 1, [2, 3])
            rot_180 = torch.rot90(image, 2, [2, 3])
            rot_270 = torch.rot90(image, 3, [2, 3])
            hor_flip = torch.flip(image, [-1])
            ver_flip = torch.flip(image, [-2])
            image = torch.cat([image, rot_90, rot_180, rot_270, hor_flip, ver_flip], dim=0)

            rot_90_cm = torch.rot90(coarsemask, 1, [2, 3])
            rot_180_cm = torch.rot90(coarsemask, 2, [2, 3])
            rot_270_cm = torch.rot90(coarsemask, 3, [2, 3])
            hor_flip_cm = torch.flip(coarsemask, [-1])
            ver_flip_cm = torch.flip(coarsemask, [-2])
            coarsemask = torch.cat([coarsemask, rot_90_cm, rot_180_cm, rot_270_cm, hor_flip_cm, ver_flip_cm], dim=0)
            # with torch.no_grad():
            #     image = torch.cat((image, coarsemask), dim=1)
            # pred = net(image)
            # F_sigmoid_cet_out, F_sigmoid_M_Net_out, F_sigmoid_ave_out, cet_out, M_Net_out, ave_out = net(image)
            # pred = F_sigmoid_ave_out
            # out, side_5, side_6, side_7, side_8 = net(image)
            pred = net(image)
            # pred = torch.log(softmax_2d(side_8) + EPS)
            #
            pred = pred[0:1] + torch.rot90(pred[1:2], 3, [2, 3]) + torch.rot90(pred[2:3], 2, [2, 3]) + torch.rot90(
                pred[3:4], 1, [2, 3]) + torch.flip(pred[4:5], [-1]) + torch.flip(pred[5:6], [-2])

            predicted = pred.argmax(1)

            test_mdice += Miou.calculate_mdice(predicted, targets, 2).item()
            # dice+=Dice(predicted,targets)
            test_miou += Miou.calculate_miou(predicted, targets, 2).item()

            test_pre += Miou.pre(predicted, targets).item()
            test_recall += Miou.recall(predicted, targets).item()
            test_F1score += Miou.F1score(predicted, targets).item()
            predict = predicted.squeeze(0)
            label = label.squeeze(0)
            label = label.cpu().numpy()
            img_np = predict.cpu().numpy()  # np.array
            predict = predicted.squeeze(0)
            img_ori = img.squeeze(0)
            img_np = predict.cpu().numpy()  # np.array
            img_ori = img_ori.cpu().numpy()
            img_ori = img_ori.transpose(2, 1, 0)
            label_224 = label
            img_224 = img_ori
            size = img_np.shape[:2]

            img_np = (img_np * 255).astype('uint8')
            label = (label * 255).astype('uint8')
            # label_224 = (label_224 * 255).astype('uint8')
            # img_224 = (img_224 * 255).astype('uint8')
            # img_np = cv2.resize(img_np, dsize=(256, 256))
            # label = cv2.resize(label, dsize=(256, 256))
            i = img_path[0].split('/')[-1]
            # print(img_path.shape)
            # img_224 = cv2.cvtColor(img_224, cv2.COLOR_BGR2RGB)

            # cv2.imwrite(os.path.join(result_path, i), img_np)

            logi = F.softmax(torch.squeeze(pred), dim=0)[1].to('cpu').numpy()
            cv2.imwrite(os.path.join(result_roc_path, i),cv2.resize((logi * 255),size[::-1], interpolation=cv2.INTER_NEAREST))
            # cv2.imwrite(os.path.join(masks_224_path, i),
            #             cv2.resize((label_224), size[::-1], interpolation=cv2.INTER_NEAREST))
            # cv2.imwrite(os.path.join(img_224_path, i), cv2.resize((img_224), size[::-1]))

        print(test_mdice / batch_idx)
        print(test_miou / batch_idx)
        print(test_pre / batch_idx)
        print(test_recall / batch_idx)
        print(test_F1score / batch_idx)


if __name__ == '__main__':
    test_data()