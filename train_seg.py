'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import os
import argparse
from tqdm import tqdm
from my_dataset.my_datasets import MyDataSet_seg, MyValDataSet_seg

import utils.utils as utils
import models_00.loss.Loss_all as Loss
import models_00.metrics.Miou_bak as Miou
import models_00.AGNET_model.AGNET as AGNET
from torch.utils import data

parser = argparse.ArgumentParser(description='Segmentation method training')
parser.add_argument('--resize', default=512, type=int, help='resize shape')
parser.add_argument('--batch_size', default=16,type=int,help='batchsize')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--start_epoch', '-s', default=0, type=int, help='start epoch')
parser.add_argument('--end_epoch', '-e', default=300, type=int, help='end epoch')
parser.add_argument('--times', '-t', default=1, type=int, help='val')
parser.add_argument('--device', default='cuda', type=str, help='use cuda')
parser.add_argument('--tb_path', type=str, default='log/Unet', help='tensorboard path')
parser.add_argument('--checkpoint', type=str, default='checkpoint/my_wbc_checkpoint/', help='checkpoint path')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
torch.backends.cudnn.enabled =True
tb_path = args.tb_path
if not os.path.exists(tb_path):
    os.mkdir(tb_path)
device = args.device # 是否使用cuda
# model_urls = {'deeplabv3plus_xception': 'models_00/pretrained_models/deeplabv3plus_xception_VOC2012_epoch46_all.pth'}

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

best_miou = 0  # best test accuracy
EPS = 1e-12
start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
times = args.times  # 验证次数
checkpoint_path = args.checkpoint + 'wbc_M_Net_CAC_with_coarse_slide.pth'
print('==> Preparing data..')

w, h = 512, 512
############# Load training and validation data
data_train_root = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/train/'
data_csv_root = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/CCE-NET/Train/coarse_mask3/'
data_train_list = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/train_img_label_seg.txt'
trainloader = data.DataLoader(MyDataSet_seg(data_train_root, data_train_list, root_path_csv=data_csv_root, crop_size=(w, h)),
                                  batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

data_val_root = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/val/'
data_csv_root = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/CCE-NET/Val/coarse_mask3/'
data_val_list = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/val_img_label_seg.txt'
valloader = data.DataLoader(MyValDataSet_seg(data_val_root, data_val_list, root_path_coarsemask=data_csv_root), batch_size=1, shuffle=False,
                                num_workers=8, pin_memory=True)

# Model
print('==> Building model..')

net = AGNET.M_Net_CAC_with_CAM_Slide(in_ch=3, out_ch=2)
# net = deeplabv3plus(num_classes=2)
print("param size = %fMB", utils.count_parameters_in_MB(net))

net = net.to(device)
# print(args.resume)
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.checkpoint), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    best_miou = checkpoint['miou']
    start_epoch = checkpoint['epoch']

criterion = Loss.CrossEntropyLoss2D().to(device)
# criterion = nn.NLLLoss2d().to(device)
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200,eta_min=1e-8)

softmax_2d = nn.Softmax2d()

############# Load pretrained weights
# pretrained_dict = torch.load(model_urls['deeplabv3plus_xception'])
# net_dict = net.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape == net_dict[k].shape)}
# net_dict.update(pretrained_dict)
# net.load_state_dict(net_dict)

net.train()
net.float()
def train_val():
    with SummaryWriter(tb_path) as write:
        train_step = 0
        for epoch in range(start_epoch, args.end_epoch):
            with tqdm(total=len(trainloader), ncols=120, ascii=True) as t:
                net.train()
                train_loss = 0
                train_miou = 0

                for batch_idx, (inputs, coarsemask, label, name) in enumerate(trainloader):
                    t.set_description("Train(Epoch{}/{})".format(epoch, args.end_epoch))
                    inputs, label, coarsemask = inputs.to(device), label.to(device), coarsemask.to(device)
                    coarsemask = coarsemask.unsqueeze(1).cuda()


                    # with torch.no_grad():
                    #     inputs = torch.cat((inputs, coarsemask), dim=1)

                    optimizer.zero_grad()
                    # out = net(inputs)
                    # label = label.long()
                    # loss = criterion(out, label)

                    label = label.long()

                    out, side_5, side_6, side_7, side_8 = net(inputs,coarsemask)
                    out = torch.log(softmax_2d(out) + EPS)
                    loss = criterion(out, label)
                    loss += criterion(torch.log(softmax_2d(side_5) + EPS), label)
                    loss += criterion(torch.log(softmax_2d(side_6) + EPS), label)
                    loss += criterion(torch.log(softmax_2d(side_7) + EPS), label)
                    loss += criterion(torch.log(softmax_2d(side_8) + EPS), label)
                    out = torch.log(softmax_2d(side_8) + EPS)

                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    predicted = out.argmax(1)
                    train_miou += Miou.calculate_miou(predicted, label, 2).item()

                    write.add_scalar('Train_loss', train_loss / (batch_idx + 1), global_step=train_step)
                    write.add_scalar('Train_Miou', 100. * (train_miou / (batch_idx + 1)), global_step=train_step)
                    train_step += 1
                    t.set_postfix(loss='{:.3f}'.format(train_loss / (batch_idx + 1)),
                                  train_miou='{:.2f}%'.format(100. * (train_miou / (batch_idx + 1))))
                    t.update(1)

                scheduler.step()
            if epoch % times == 0:
                global best_miou
                net.eval()
                val_loss = 0
                val_miou = 0
                with torch.no_grad():
                    with tqdm(total=len(valloader), ncols=120, ascii=True) as t:
                        for batch_idx, (inputs, coarsemask, label, name) in enumerate(valloader):
                            t.set_description("Val(Epoch {}/{})".format(epoch, args.end_epoch))

                            inputs, label, coarsemask = inputs.cuda(), label.cuda(), coarsemask.cuda()
                            coarsemask = coarsemask.unsqueeze(1).cuda()

                            label = label.long()

                            # with torch.no_grad():
                            #     inputs = torch.cat((inputs, coarsemask),dim=1)

                            # outputs = net(inputs)
                            #
                            # loss = criterion(outputs, label)

                            label = label.long()

                            out, side_5, side_6, side_7, side_8 = net(inputs,coarsemask)
                            out = torch.log(softmax_2d(out) + EPS)
                            loss = criterion(out, label)
                            loss += criterion(torch.log(softmax_2d(side_5) + EPS), label)
                            loss += criterion(torch.log(softmax_2d(side_6) + EPS), label)
                            loss += criterion(torch.log(softmax_2d(side_7) + EPS), label)
                            loss += criterion(torch.log(softmax_2d(side_8) + EPS), label)
                            out = torch.log(softmax_2d(side_8) + EPS)


                            val_loss += loss.item()
                            predicted = out.argmax(1)
                            val_miou += Miou.calculate_miou(predicted, label, 2).item()


                            t.set_postfix(loss='{:.3f}'.format(val_loss / (batch_idx + 1)),
                                          val_miou='{:.2f}%'.format(100. * (val_miou / (batch_idx + 1))))
                            t.update(1)
                        write.add_scalar('Val_loss', val_loss / (batch_idx + 1), global_step=train_step)
                        write.add_scalar('Val_miou', 100. * (val_miou / (batch_idx + 1)), global_step=train_step)
                        # Save checkpoint.
                    val_miou = 100. * (val_miou / (batch_idx + 1))
                    if val_miou > best_miou:
                        print('Saving..')
                        state = {
                            'net': net.state_dict(),
                            'miou': val_miou,
                            'epoch': epoch,
                        }
                        if not os.path.isdir(args.checkpoint):
                            os.mkdir(args.checkpoint)
                        torch.save(state, checkpoint_path)
                        best_miou = val_miou



if __name__ == '__main__':
    train_val()
