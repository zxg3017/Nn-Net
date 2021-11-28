'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
import argparse
from tqdm import tqdm
from my_dataset.my_datasets import MyDataSet_cls,MyValDataSet_cls

import utils.utils as utils
from torch.utils import data
import models_00.AGNET_model.n_ClsNet as n_ClsNet

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--imgs_train_path', '-it', type=str,
                    default='/mnt/ai2019/zxg_FZU/dataset/semi_data/train/label_img/Images',
                    help='imgs train data path.')
parser.add_argument('--labels_train_path', '-lt', type=str,
                    default='/mnt/ai2019/zxg_FZU/dataset/semi_data/train/label_img/Annotation',
                    help='labels train data path.')
parser.add_argument('--imgs_val_path', '-iv', type=str, default='/mnt/ai2019/zxg_FZU/dataset/semi_data/val/Images',
                    help='imgs val data path.')
parser.add_argument('--labels_val_path', '-lv', type=str,
                    default='/mnt/ai2019/zxg_FZU/dataset/semi_data/val/Annotation', help='labels val data path.')
parser.add_argument('--resize', default=400, type=int, help='resize shape')
parser.add_argument('--batch_size', default=8, type=int, help='batchsize')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--start_epoch', '-s', default=0, type=int, help='start epoch')
parser.add_argument('--end_epoch', '-e', default=300, type=int, help='end epoch')
parser.add_argument('--times', '-t', default=1, type=int, help='val')
parser.add_argument('--device', default='cuda', type=str, help='use cuda')
parser.add_argument('--tb_path', type=str, default='log/Unet', help='tensorboard path')
parser.add_argument('--checkpoint', type=str, default='checkpoint/my_model_cls/', help='checkpoint path')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
torch.backends.cudnn.enabled = True
tb_path = args.tb_path
if not os.path.exists(tb_path):
    os.mkdir(tb_path)
device = args.device  # 是否使用cuda

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

best_miou = 0  # best test accuracy
EPS = 1e-12
start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
times = args.times  # 验证次数
checkpoint_path = args.checkpoint + 'my_resnet101.pth'
# Data
print('==> Preparing data..')

w, h = 512, 512
############# Load training and validation data
data_train_root = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/train/'
data_csv_root = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/CCE-NET/Train/coarse_mask3/'
data_train_list = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/train_img_label_seg.txt'
trainloader = data.DataLoader(
    MyDataSet_cls(data_train_root, data_csv_root, data_train_list),
    batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

data_val_root = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/val/'
data_csv_root = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/CCE-NET/Val/coarse_mask3/'
data_val_list = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/val_img_label_seg.txt'
valloader = data.DataLoader(MyValDataSet_cls(data_val_root, data_csv_root, data_val_list, ),
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True, drop_last=True)
# Model
print('==> Building model..')
# net = resnet101()
net =n_ClsNet.M_Net_CAC_with_CAM_Slide_CLS (3,5)

print("param size = %fMB", utils.count_parameters_in_MB(net))

# print(args.resume)
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.checkpoint), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# criterion = Loss.CrossEntropyLoss2D().to(device)
# criterion = Loss.FocalLoss().to(device)
loss_function = nn.CrossEntropyLoss()

# criterion = nn.NLLLoss2d().to(device)
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-8)

softmax_2d = nn.Softmax2d()

############# Load pretrained weights
model_weight_path = "/mnt/ai2019/zxg_FZU/models_00/resnet/resnet101.pth"
# assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
# net.load_state_dict(torch.load(model_weight_path, map_location=device))
# for param in net.parameters():
#     param.requires_grad = False

# change fc layer structure
in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 5)
net.to(device)


val_num = len(trainloader)
best_acc = 0.0
def train_val():
    with SummaryWriter(tb_path) as write:
        train_step = 0
        for epoch in range(start_epoch, args.end_epoch):
            with tqdm(total=len(trainloader), ncols=120, ascii=True) as t:
                net.train()
                train_loss = 0
                train_acc = 0

                for batch_idx, (inputs, coarsemask, label, name) in enumerate(trainloader):
                    t.set_description("Train(Epoch{}/{})".format(epoch, args.end_epoch))
                    inputs, label = inputs.to(device), label.to(device)

                    optimizer.zero_grad()
                    out = net(inputs)
                    # label = label.long()
                    # predicted = torch.max(out, dim=1)
                    # loss = criterion(predicted_y, label)
                    loss = loss_function(out, label.long())

                    loss.backward()

                    optimizer.step()
                    train_loss += loss.item()
                    predicted = out.argmax(1)
                    train_acc += torch.eq(predicted, label.to(device)).sum().item()

                    write.add_scalar('Train_loss', train_loss / (batch_idx + 1), global_step=train_step)
                    write.add_scalar('Train_acc', 100. * (train_acc / ((batch_idx + 1)*args.batch_size)), global_step=train_step)
                    train_step += 1
                    t.set_postfix(loss='{:.3f}'.format(train_loss / (batch_idx + 1)),
                                  train_acc='{:.2f}%'.format(100. * (train_acc / ((batch_idx + 1)*args.batch_size))))
                    t.update(1)

                scheduler.step()
            if epoch % times == 0:
                global best_acc
                net.eval()
                val_loss = 0
                val_acc = 0
                with torch.no_grad():
                    with tqdm(total=len(valloader), ncols=120, ascii=True) as t:
                        for batch_idx, (inputs, coarsemask, label, name) in enumerate(valloader):
                            t.set_description("Val(Epoch {}/{})".format(epoch, args.end_epoch))

                            inputs, label = inputs.cuda(), label.cuda()
                            coarsemask = coarsemask.unsqueeze(1).cuda()

                            # label = label.long()

                            # with torch.no_grad():
                            #     input_csv = torch.cat((inputs, coarsemask), dim=1)

                            outputs = net(inputs)

                            loss = loss_function(outputs, label.long())

                            val_loss += loss.item()
                            # predicted = outputs.argmax(1)
                            predicted = out.argmax(1)

                            val_acc += torch.eq(predicted, label.to(device)).sum().item()

                            # val_miou += Miou.calculate_miou(predicted, label, 2).item()

                            t.set_postfix(loss='{:.3f}'.format(val_loss / (batch_idx + 1)),
                                          val_acc='{:.2f}%'.format(100. * (val_acc / ((batch_idx + 1)*args.batch_size))))
                            t.update(1)
                        write.add_scalar('Val_loss', val_loss / (batch_idx + 1), global_step=train_step)
                        write.add_scalar('Val_acc', 100. * (val_acc / ((batch_idx + 1)*args.batch_size)), global_step=train_step)
                        # Save checkpoint.
                    val_acc = 100. * (val_acc / (batch_idx + 1))
                    if val_acc > best_acc:
                        print('Saving..')
                        state = {
                            'net': net.state_dict(),
                            'best_acc': val_acc,
                            'epoch': epoch,
                        }
                        if not os.path.isdir(args.checkpoint):
                            os.mkdir(args.checkpoint)
                        torch.save(state, checkpoint_path)
                        best_acc = val_acc


if __name__ == '__main__':
    train_val()
