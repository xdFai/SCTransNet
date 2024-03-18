import argparse
import time
import os
import cv2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import *
from metrics import *
from utils import *
import model.Config as config
from torch.utils.tensorboard import SummaryWriter
from model.SCTransNet import SCTransNet as SCTransNet

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
parser.add_argument("--model_names", default=['SCTransNet'], type=list, help="'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'RISTDnet'")
parser.add_argument("--dataset_names", default=['SIRST3'], type=list)
# SIRST3： NUAA NUDT IRSTD-1K
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: AdamW, Adam, Adagrad, SGD")
parser.add_argument("--epochs", default=1000, type=int, help="optimizer name: AdamW, Adam, Adagrad, SGD")
parser.add_argument("--begin_test", default=500, type=int)
parser.add_argument("--every_test", default=1, type=int)
parser.add_argument("--every_save_pth", default=1000, type=int)
parser.add_argument("--every_print", default=10, type=int)
parser.add_argument("--dataset_dir", default=r'./datasets')
parser.add_argument("--batchSize", type=int, default=16, help="Training batch sizse")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--save", default=r'./log', type=str, help="Save path of checkpoints")
parser.add_argument("--log_dir", type=str, default="./otherlogs/SCTransNet", help='path of log files')
parser.add_argument("--img_norm_cfg", default=None, type=dict)
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
parser.add_argument("--resume", default=False, type=list, help="Resume from exisiting checkpoints (default: None)")

global opt
opt = parser.parse_args()

seed_pytorch(opt.seed)

config_vit = config.get_SCTrans_config()


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def train():
    train_set = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize,
                               img_norm_cfg=opt.img_norm_cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    net = Net(model_name=opt.model_name, mode='train').cuda()
    net.apply(weights_init_kaiming)
    net.train()

    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)
    writer = SummaryWriter(opt.log_dir)

    if opt.resume:
        # for resume_pth in opt.resume:
        #     if opt.dataset_name in resume_pth and opt.model_name in resume_pth:
        ckpt = torch.load('XX\\UCT04_best.pth.tar')
        net.load_state_dict(ckpt['state_dict'])
        epoch_state = ckpt['epoch']
        total_loss_list = ckpt['total_loss']
        # for i in range(len(opt.scheduler_settings['step'])):
        #     opt.scheduler_settings['step'][i] = opt.scheduler_settings['step'][i] - ckpt['epoch']

    ### Default settings of SCTransNet
    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 0.001}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': opt.epochs, 'eta_min': 1e-5, 'last_epoch': -1}

    ### Default settings of DNANet
    if opt.optimizer_name == 'Adagrad':
        opt.optimizer_settings = {'lr': 0.05}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': opt.epochs, 'min_lr': 1e-5}

    ### Default settings of EGEUNet
    if opt.optimizer_name == 'AdamW':
        opt.optimizer_settings = {'lr': 0.001, 'betas': (0.9, 0.999), "eps": 1e-8, "weight_decay": 1e-2,
                                  "amsgrad": False}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': opt.epochs, 'T_max': 50, 'eta_min': 1e-5, 'last_epoch': -1}

    opt.nEpochs = opt.scheduler_settings['epochs']

    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings,
                                         opt.scheduler_settings)


    for idx_epoch in range(epoch_state, opt.nEpochs):

        results1 = (0, 0)
        results2 = (0, 0)
        for idx_iter, (img, gt_mask) in enumerate(train_loader):
            img, gt_mask = Variable(img).cuda(), Variable(gt_mask).cuda()
            if img.shape[0] == 1:
                continue
            preds = net.forward(img)
            loss = net.loss(preds, gt_mask)
            total_loss_epoch.append(loss.detach().cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if (idx_epoch + 1) % opt.every_print == 0:
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f, lr---%f,'
                  % (idx_epoch + 1, total_loss_list[-1], scheduler.get_last_lr()[0]))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n'
                        % (idx_epoch + 1, total_loss_list[-1]))
            total_loss_epoch = []
            # Log the scalar values
            writer.add_scalar('loss', total_loss_list[-1], idx_epoch + 1)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], idx_epoch + 1)

        # 500
        if (idx_epoch + 1) >= opt.begin_test and (idx_epoch + 1) % opt.every_test == 0:
            test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, img_norm_cfg=opt.img_norm_cfg)
            test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
            net.eval()
            with torch.no_grad():
                eval_mIoU = mIoU()
                eval_PD_FA = PD_FA()
                test_loss = []
                for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
                    img = Variable(img).cuda()
                    pred = net.forward(img)
                    if isinstance(pred, tuple):
                        pred = pred[-1]
                    elif isinstance(pred, list):
                        pred = pred[-1]
                    else:
                        pred = pred
                    pred = pred[:, :, :size[0], :size[1]]
                    gt_mask = gt_mask[:, :, :size[0], :size[1]]
                    # if pred.size() != gt_mask.size():
                    #     print('1111')
                    loss = net.loss(pred, gt_mask.cuda())
                    test_loss.append(loss.detach().cpu())
                    eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask.cpu())
                    eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size)
                test_loss.append(float(np.array(test_loss).mean()))
                results1 = eval_mIoU.get()
                results2 = eval_PD_FA.get()
                writer.add_scalar('mIOU', results1[-1], idx_epoch + 1)
                writer.add_scalar('testloss', test_loss[-1], idx_epoch + 1)


        if (idx_epoch + 1) % opt.every_save_pth == 0:
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
            }, save_pth)
            test(save_pth)

        if idx_epoch == 0:
            best_mIOU = results1
            best_Pd = results2

        if results1[1] > best_mIOU[1]:
            best_mIOU = results1
            best_Pd = results2
            print('------save the best model epoch', opt.model_name,'_%d ------' % (idx_epoch + 1))
            opt.f.write("the best model epoch \t" + str(idx_epoch + 1) + '\n')
            print("pixAcc, mIoU:\t" + str(best_mIOU))
            print("testloss:\t" + str(test_loss[-1]))
            print("PD, FA:\t" + str(best_Pd))

            opt.f.write("pixAcc, mIoU:\t" + str(best_mIOU) + '\n')
            opt.f.write("PD, FA:\t" + str(best_Pd) + '\n')
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '_' + 'best' + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
            }, save_pth)

        # last epoch
        if (idx_epoch + 1) == opt.nEpochs and (idx_epoch + 1) % opt.every_save_pth != 0:
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
            }, save_pth)
            test(save_pth)


def test(save_pth):
    test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, img_norm_cfg=opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    net = Net(model_name=opt.model_name, mode='test').cuda()
    ckpt = torch.load(save_pth)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    with torch.no_grad():
        eval_mIoU = mIoU()
        eval_PD_FA = PD_FA()
        test_loss_a = []
        for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
            img = Variable(img).cuda()
            pred = net.forward(img)
            if pred.size() != gt_mask.size():
                print('1111')
            pred = pred[:, :, :size[0], :size[1]]
            gt_mask = gt_mask[:, :, :size[0], :size[1]]
            loss = net.loss(pred, gt_mask.cuda())
            test_loss_a.append(loss.detach().cpu())
            eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask.cpu())
            eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size)

        test_loss_a.append(float(np.array(test_loss_a).mean()))
        results1 = eval_mIoU.get()
        results2 = eval_PD_FA.get()

        print('== == == == == == == ', opt.model_name, ' == == == == == == ==')
        print("pixAcc, mIoU:\t" + str(results1))
        print("testloss:\t" + str(test_loss_a[-1]))
        print("PD, FA:\t" + str(results2))
        opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
        opt.f.write("PD, FA:\t" + str(results2) + '\n')


def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path


class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        # ************************************************loss*************************************************#
        self.cal_loss = nn.BCELoss(size_average=True)
        if model_name == 'SCTransNet':
            if mode == 'train':
                self.model = SCTransNet(config_vit, mode='train', deepsuper=True)
            else:
                self.model = SCTransNet(config_vit, mode='test', deepsuper=True)
    def forward(self, img):
        return self.model(img)

    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)

        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss


if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ',
                                                                                                                 '_').replace(
                ':', '_') + '.txt', 'w')
            print(opt.dataset_name + '\t' + opt.model_name)
            train()
            print('\n')
            opt.f.close()
