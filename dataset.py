from utils import *
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//', '/')).convert(
                'I')  # read image base on version ”I“
            # img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//','/'))
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.bmp').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.bmp').replace('//', '/'))
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)  # convert PIL to numpy  and  normalize
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        # rnd_bn = np.random.normal(0, 0.03)#0.03
        # img += rnd_bn
        #
        # minm = img.min()
        # rng = img.max() - minm
        # gamma = np.random.uniform(0.5, 1.6)
        # x=((img - minm) / rng)
        # img = np.power(x, gamma)
        # img = img * rng + minm

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)  # 把短的一边先pad至256 把长的一边 随机裁出256  输出 256 256

        img_patch, mask_patch = self.tranform(img_patch, mask_patch)  # 数据翻转增强
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]  # 升维
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))  # numpy 转tensor
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))  # numpy 转tensor
        return img_patch, mask_patch

    def __len__(self):
        return len(self.train_list)

class TrainSetLoader02(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//', '/')).convert(
                'I')  # read image base on version ”I“
            # img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//','/'))
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.bmp').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.bmp').replace('//', '/'))
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)  # convert PIL to numpy  and  normalize
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        rnd_bn = np.random.normal(0, 0.03)#0.03
        img += rnd_bn
        #
        # minm = img.min()
        # rng = img.max() - minm
        # gamma = np.random.uniform(0.5, 1.6)
        # x=((img - minm) / rng)
        # img = np.power(x, gamma)
        # img = img * rng + minm

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)  # 把短的一边先pad至256 把长的一边 随机裁出256  输出 256 256

        img_patch, mask_patch = self.tranform(img_patch, mask_patch)  # 数据翻转增强
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]  # 升维
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))  # numpy 转tensor
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))  # numpy 转tensor
        return img_patch, mask_patch

    def __len__(self):
        return len(self.train_list)


class TrainSetLoader03(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//', '/')).convert(
                'I')  # read image base on version ”I“
            # img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//','/'))
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.bmp').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.bmp').replace('//', '/'))
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)  # convert PIL to numpy  and  normalize
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        # rnd_bn = np.random.normal(0, 0.03)#0.03
        # img += rnd_bn

        minm = img.min()
        rng = img.max() - minm
        gamma = np.random.uniform(0.5, 1.6)
        x=((img - minm) / rng)
        img = np.power(x, gamma)
        img = img * rng + minm

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)  # 把短的一边先pad至256 把长的一边 随机裁出256  输出 256 256

        img_patch, mask_patch = self.tranform(img_patch, mask_patch)  # 数据翻转增强
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]  # 升维
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))  # numpy 转tensor
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))  # numpy 转tensor
        return img_patch, mask_patch

    def __len__(self):
        return len(self.train_list)


class TrainSetLoader04(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//', '/')).convert(
                'I')  # read image base on version ”I“
            # img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//','/'))
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.bmp').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.bmp').replace('//', '/'))
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)  # convert PIL to numpy  and  normalize
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        rnd_bn = np.random.normal(0, 0.03)#0.03
        img += rnd_bn

        minm = img.min()
        rng = img.max() - minm
        gamma = np.random.uniform(0.5, 1.6)
        x=((img - minm) / rng)
        img = np.power(x, gamma)
        img = img * rng + minm

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)  # 把短的一边先pad至256 把长的一边 随机裁出256  输出 256 256

        img_patch, mask_patch = self.tranform(img_patch, mask_patch)  # 数据翻转增强
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]  # 升维
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))  # numpy 转tensor
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))  # numpy 转tensor
        return img_patch, mask_patch

    def __len__(self):
        return len(self.train_list)

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
        # with open(r'D:\05TGARS\Upload\datasets\SIRST3\img_idx\val.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.bmp').replace('//', '/'))

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        # if mask.shape == (416,608):
        #     print('111')
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape

        img = PadImg(img)
        mask = PadImg(mask)

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        if img.size() != mask.size():
            print('111')
        return img, mask, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)


class EvalSetLoader(Dataset):
    def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name, model_name):
        super(EvalSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.mask_pred_dir = mask_pred_dir
        self.test_dataset_name = test_dataset_name
        self.model_name = model_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()

    def __getitem__(self, idx):
        mask_pred = Image.open(
            (self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/' + self.test_list[idx] + '.png').replace('//', '/'))
        mask_gt = Image.open(self.dataset_dir + '/masks/' + self.test_list[idx] + '.png')

        mask_pred = np.array(mask_pred, dtype=np.float32) / 255.0
        mask_gt = np.array(mask_gt, dtype=np.float32) / 255.0

        if len(mask_pred.shape) == 3:
            mask_pred = mask_pred[:, :, 0]

        h, w = mask_pred.shape

        mask_pred, mask_gt = mask_pred[np.newaxis, :], mask_gt[np.newaxis, :]

        mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
        mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
        return mask_pred, mask_gt, [h, w]

    def __len__(self):
        return len(self.test_list)


class augumentation(object):
    def __call__(self, input, target):
        if random.random() < 0.5:  # 水平反转
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:  # 垂直反转
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random() < 0.5:  # 转置反转
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input, target
