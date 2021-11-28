import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util

class CreateDataset(data.Dataset):
    def __init__(self, img_paths, label_paths, resize, phase, aug=False):
        super(CreateDataset, self).__init__()
        self.phase = phase
        self.resize = resize
        self.aug = aug

        self.paths_imgs = util.get_image_paths(img_paths)
        assert self.paths_imgs, 'Error: imgs paths are empty.'

        self.paths_label = util.get_image_paths(label_paths)
        assert self.paths_label, 'Error: paths_label are empty.'

    def __getitem__(self, index):

        img_path = self.paths_imgs[index]

        img = util.read_img(img_path)  # h w c

        lable_path = self.paths_label[index]

        label = util.read_nodule_label(lable_path)  # h w

        img = cv2.resize(img, (self.resize, self.resize))
        label = cv2.resize(label, (self.resize, self.resize))

        if self.phase == 'train':
            if self.aug:
                img, label= util.augment([img, np.expand_dims(label, axis=2)], hflip=True, rot=True)

                label = label.squeeze(2) #HW
                img = img[:, :, [2, 1, 0]] # bgr -> rgb

        elif self.phase == 'val':
            if img.shape[2] == 3:
                img = img[:, :, [2, 1, 0]]

            # if label.shape[2] == 3:
            #     label = label[:,:, 1]
        # print(label.shape)
        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()

        label = torch.from_numpy(np.ascontiguousarray(label)).long()


        return img, label, img_path

    def __len__(self):
        return len(self.paths_imgs)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data_train = CreateDataset('/mnt/ai2019/tutu/segmentation/test/images',
                               '/mnt/ai2019/tutu/segmentation/test/masks', phase='val', resize=512, aug=False)
    data_v = DataLoader(data_train, batch_size=1, shuffle=True, num_workers=5, pin_memory=True)
    for i, j, img_path in data_v:
        print(i.shape)
        print(j.shape)

