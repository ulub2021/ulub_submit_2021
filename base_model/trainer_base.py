import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter # pytorch 1.14 or above
import torchvision.models as models
import numpy as np

from tqdm import tqdm
from utils import logger_setting

from adamp import AdamP, SGDP

class Trainer(object):
    def __init__(self, option, logger=None):
        super(Trainer, self).__init__()

        self.option = option
        self.writer = SummaryWriter(os.path.join(self.option.save_dir, self.option.exp_name, 'progress_log'))

        self._build_model()
        self._set_optimizer()

        if logger is not None:
            self.logger = logger
        else:
            self.logger = logger_setting(option.exp_name, option.save_dir, option.debug)

    def freeze_all(self, model_params):
        for param in model_params:
            param.requires_grad = False

    def requires_grad(self, layer):
        "Determines whether 'layer' requires gradients"
        ps = list(layer.parameters())
        if not ps: return None
        return ps[0].requires_grad

    def _build_model(self):
        if self.option.model == 'resnet18':
            if self.option.imagenet_pretrain:
                print('Pretrained resnet18')
                self.net = models.resnet18(pretrained=True)
                if not self.option.data == 'imagenet_16class':
                    self.net.fc = torch.nn.Linear(512, self.option.n_class)
            else:
                print('Start from scratch renset18')
                self.net = models.resnet18(pretrained=False, num_classes=self.option.n_class)

        elif self.option.model == 'vgg11':
            if self.option.imagenet_pretrain:
                print('Pretrained vgg11')
                self.net = models.vgg11(pretrained=True)
                if not self.option.data == 'imagenet_16class':
                   self.net.classifier[6] = torch.nn.Linear(4096, self.option.n_class)
            else:
                print('Start from scratch vgg11')
                self.net = models.vgg11(pretrained=False, num_classes=self.option.n_class)

        self.loss = nn.CrossEntropyLoss(ignore_index=255)

        if self.option.cuda:
            self.net.cuda()
            self.loss.cuda()

    def _set_optimizer(self):
        if self.option.optim == 'AdamP':
            self.optim = AdamP(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.option.lr,
                                    betas=(0.9, 0.999), weight_decay=self.option.weight_decay)
        elif self.option.optim == 'SGDP':
            self.optim = SGDP(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.option.lr,
                                  weight_decay=self.option.weight_decay, momentum=0.9, nesterov=True)
        elif self.option.optim == 'SGD':
            self.optim = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.option.lr,
                                        weight_decay=self.option.weight_decay, momentum=0.9, nesterov=True)
        elif self.option.optim == 'Adam':
            self.optim = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                         lr=self.option.lr, weight_decay=self.option.weight_decay)
        else:
            self.logger.error("wrong optimizer")
        lr_lambda = lambda step: self.option.lr_decay_rate ** (step // self.option.lr_decay_period)
        if self.option.lr_scheduler == 'step':
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda, last_epoch=-1)
        elif self.option.lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=self.option.max_step, eta_min=0)

    @staticmethod
    def _weights_init_xavier(m):
        classname = m.__class__.__name__
        if classname == 'BasicConv2d' or classname == 'ConvBNReLU':
            pass
        elif classname.find('Conv') != -1 or classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight.data, gain=1.0)

    def _initialization(self):
        if self.option.is_train:
            if self.option.imagenet_pretrain:
                if 'classifier' in [n for n, p in self.net.named_children()]:
                    self.net.classifier[-1].apply(self._weights_init_xavier)
                elif 'fc' in [n for n, p in self.net.named_children()]:
                    self.net.fc.apply(self._weights_init_xavier)
            else:
                self.net.apply(self._weights_init_xavier)

            if self.option.use_pretrain:
                if self.option.checkpoint is not None:
                    self._load_model()
                else:
                    print("no prtrained model")

    def _mode_setting(self, is_train=True):
        if is_train:
            self.net.train()
        else:
            self.net.eval()

    def _train_step(self, data_loader, step):
        loss_sum = 0.
        total_num_correct = 0.
        total_num_test = 0.
        for i, cur_data in enumerate(tqdm(data_loader, leave=False)):
            if self.option.data == 'imagenet':
                images, labels, bias_labels = cur_data
            else:
                images, labels = cur_data
            images = self._get_variable(images)
            labels = self._get_variable(labels)

            pred_label = self.net(images)

            total_num_correct += self._num_correct(pred_label, labels, topk=1).data
            batch_size = images.shape[0]
            total_num_test += batch_size

            loss = self.loss(pred_label, torch.squeeze(labels))
            loss_sum += loss
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.writer.add_scalar('Loss/train_step', loss / batch_size, step * len(data_loader) + i)
            self.writer.add_scalar('Accuracy/train_step',
                                   self._num_correct(pred_label, labels, topk=1).data / batch_size,
                                   step * len(data_loader) + i)
        avg_acc = float(total_num_correct) / total_num_test
        msg = f"[TRAIN] LOSS  {loss_sum / len(data_loader)}, ACCURACY : {avg_acc}, LR : {self.optim.param_groups[0]['lr']}"

        self.writer.add_scalars('Loss/epoch', {'train': loss_sum / len(data_loader)}, step)
        self.writer.add_scalars('Accuracy/epoch', {'train': avg_acc}, step)
        self.writer.add_scalar('Learning rate', self.optim.param_groups[0]['lr'], step)

        self.logger.info(msg)

    def _validate(self, data_loader, step=0, valid_type=''):
        self._mode_setting(is_train=False)

        if not self.option.is_train:
            print('not in training process')
            self._initialization()
            if self.option.checkpoint is not None:
                self._load_model()
            else:
                print("No trained model")
                sys.exit()

        total_num_correct = 0.
        total_num_test = 0.
        total_loss = 0.
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(data_loader, leave=False)):
                images = self._get_variable(images)
                labels = self._get_variable(labels)

                pred_label = self.net(images)

                loss = self.loss(pred_label, torch.squeeze(labels))

                batch_size = images.shape[0]
                total_num_correct += self._num_correct(pred_label, labels, topk=1).data
                total_loss += loss.data * batch_size
                total_num_test += batch_size

        avg_loss = total_loss / total_num_test
        avg_acc = float(total_num_correct) / total_num_test
        msg = f"[EVALUATION({valid_type})] step {step}, LOSS {avg_loss}, ACCURACY : {avg_acc}"
        self.logger.info(msg)

        self.writer.add_scalars('Loss/epoch', {f'valid_{valid_type}': avg_loss}, step)
        self.writer.add_scalars('Accuracy/epoch', {f'valid_{valid_type}': avg_acc}, step)

    def _validate_imagenet_16class(self, data_loader, step=0, valid_type=''):
        self._mode_setting(is_train=False)

        if not self.option.is_train:
            print('not in training process')
            self._initialization()
            if self.option.checkpoint is not None:
                self._load_model()
            else:
                print("No trained model")
                sys.exit()

        PtoC = probs_to_16class()

        total_num_correct = 0.
        total_num_test = 0.
        total_loss = 0.
        category_table = {"knife":0, "keyboard":1, "elephant":2, "bicycle":3, "airplane":4,
                                  "clock":5, "oven":6, "chair":7, "bear":8, "boat":9, "cat":10,
                                  "bottle":11, "truck":12, "car":13, "bird":14, "dog":15}
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(data_loader, leave=False)):
                images = self._get_variable(images)
                labels = self._get_variable(labels)

                pred_label = self.net(images)
                pred_label = torch.nn.Softmax()(pred_label)
                pred_16class = np.zeros(pred_label.size()[0], dtype=np.int)
                for i in range(pred_label.size()[0]):
                    pred_16class[i] = category_table[PtoC.probabilities_to_decision(pred_label.cpu().numpy()[i])]

                batch_size = images.shape[0]
                num_correct = np.sum(labels.cpu().numpy() == pred_16class)
                total_num_correct += num_correct
                total_num_test += batch_size

        avg_loss = total_loss / total_num_test
        avg_acc = float(total_num_correct) / total_num_test
        msg = f"[EVALUATION - {valid_type}] step {step}, LOSS {avg_loss}, ACCURACY : {avg_acc}"
        self.logger.info(msg)

        self.writer.add_scalars('Loss/epoch', {f'valid_{valid_type}': avg_loss}, step)
        self.writer.add_scalars('Accuracy/epoch', {f'valid_{valid_type}': avg_acc}, step)

    def imagenet_unbiased_accuracy(self, outputs, labels, cluster_labels,
                                   num_correct, num_instance,
                                   num_cluster_repeat=3):
        for j in range(num_cluster_repeat):
            for i in range(outputs.size(0)):
                output = outputs[i]
                label = labels[i]
                cluster_label = cluster_labels[j][i]

                _, pred = output.topk(1, 0, largest=True, sorted=True)
                correct = pred.eq(label).view(-1).float()

                num_correct[j][label][cluster_label] += correct.item()
                num_instance[j][label][cluster_label] += 1

        return num_correct, num_instance

    def n_correct(self, pred, labels):
        _, predicted = torch.max(pred.data, 1)
        # print(predicted, labels)
        n_correct = (predicted == labels).sum().item()
        return n_correct

    def _validate_imagenet(self, data_loader, step=0, valid_type='',
                        num_clusters=9,
                        num_cluster_repeat=3):
        self._mode_setting(is_train=False)

        if not self.option.is_train:
            print('not in training process')
            self._initialization()
            if self.option.checkpoint is not None:
                self._load_model()
            else:
                print("No trained model")
                sys.exit()

        total_num_test = 0.
        total_loss = 0.

        total = 0
        f_correct = 0
        num_correct = [np.zeros([self.option.n_class, num_clusters]) for _ in range(num_cluster_repeat)]
        num_instance = [np.zeros([self.option.n_class, num_clusters]) for _ in range(num_cluster_repeat)]

        with torch.no_grad():
            for i, (images, labels, bias_labels) in enumerate(tqdm(data_loader, leave=False)):
                images = self._get_variable(images)
                labels = self._get_variable(labels)

                batch_size = labels.size(0)
                total += batch_size

                pred = self.net(images)

                if valid_type == 'unbiased':
                    num_correct, num_instance = self.imagenet_unbiased_accuracy(pred.data, labels, bias_labels,
                                                                                num_correct, num_instance,
                                                                                num_cluster_repeat)
                else:
                    f_correct += self.n_correct(pred, labels)

                loss = self.loss(pred, torch.squeeze(labels))
                total_loss += loss.data * batch_size

        if valid_type == 'unbiased':
            for k in range(num_cluster_repeat):
                x, y = [], []
                _num_correct, _num_instance = num_correct[k].flatten(), num_instance[k].flatten()
                for i in range(_num_correct.shape[0]):
                    __num_correct, __num_instance = _num_correct[i], _num_instance[i]
                    if __num_instance >= 10:
                        x.append(__num_instance)
                        y.append(__num_correct / __num_instance)
                f_correct += sum(y) / len(x)

            avg_acc = f_correct / num_cluster_repeat
            print(f_correct, num_cluster_repeat)
        else:
            avg_acc = f_correct / total
            print(f_correct, total)
        avg_loss = total_loss / total_num_test
        msg = f"[EVALUATION({valid_type})] step {step}, LOSS {avg_loss}, ACCURACY : {avg_acc}"
        self.logger.info(msg)

        self.writer.add_scalars('Loss/epoch', {f'valid_{valid_type}': avg_loss}, step)
        self.writer.add_scalars('Accuracy/epoch', {f'valid_{valid_type}': avg_acc}, step)

    def _num_correct(self, outputs, labels, topk=1):
        _, preds = outputs.topk(k=topk, dim=1)
        preds = preds.t()
        correct = preds.eq(labels.view(1, -1).expand_as(preds))
        correct = correct.view(-1).sum()
        return correct

    def _accuracy(self, outputs, labels):
        batch_size = labels.size(0)
        _, preds = outputs.topk(k=1, dim=1)
        preds = preds.t()
        correct = preds.eq(labels.view(1, -1).expand_as(preds))
        correct = correct.view(-1).float().sum(0, keepdim=True)
        accuracy = correct.mul_(100.0 / batch_size)
        return accuracy

    def _save_model(self, step):
        torch.save({
            'step': step,
            'optim_state_dict': self.optim.state_dict(),
            'net_state_dict': self.net.state_dict()
        }, os.path.join(self.option.save_dir, self.option.exp_name, f'checkpoint_step_{step}.pth'))
        print('checkpoint saved. step : %d' % step)

    def _load_model(self):
        ckpt = torch.load(self.option.checkpoint)

        self.net.load_state_dict(ckpt['net_state_dict'])
        # self.optim.load_state_dict(ckpt['optim_state_dict'])

    def train(self, train_loader, val_loaders=None, val_types=None):
        # if not self.option.use_pretrain:
        self._initialization()
        if self.option.checkpoint is not None:
            self._load_model()

        start_epoch = 0
        for step in range(start_epoch, self.option.max_step):
            self._mode_setting(is_train=True)

            self._train_step(train_loader, step)
            self.scheduler.step()

            if step == 1 or step % self.option.save_step == 0 or step == (self.option.max_step - 1):
                for val_type, val_loader in zip(val_types, val_loaders):
                    if self.option.data == 'imagenet':
                        print(val_type)
                        self._validate_imagenet(val_loader, step, valid_type=val_type)
                    else:
                        self._validate(val_loader, step, valid_type=val_type)
                self._save_model(step)

    def _get_variable(self, inputs):
        if self.option.cuda:
            return Variable(inputs.cuda())
        return Variable(inputs)

