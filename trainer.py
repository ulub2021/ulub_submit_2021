# -*- coding: utf-8 -*-
import torch, os, sys
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter # pytorch 1.14 or above
import torchvision.models as models

import model.ubnet_vgg11 as model_vgg11
import model.ubnet_resnet18 as model_resnet18
import numpy as np

from tqdm import tqdm
from adamp import AdamP, SGDP

class Trainer(object):
    def __init__(self, option, logger):
        self.option = option
        self.logger = logger

        self._build_model()
        self._set_optimizer()
        self.writer = SummaryWriter(os.path.join(self.option.save_dir, self.option.exp_name, 'progress_log'))

    def _build_model(self):
        self.logger.info(f"imagenet pretrain: {self.option.imagenet_pretrain}")

        if self.option.model == 'vgg11':
            if self.option.imagenet_pretrain:
                self.logger.info("ImageNet Pre-trained")
                self.net = models.vgg11(pretrained = True)
                self.net.classifier[6] = nn.Linear(4096, self.option.n_class)
            else:
                self.net = models.vgg11()
                self.net.classifier[6] = nn.Linear(4096, self.option.n_class)

        elif self.option.model == 'resnet18':
            if self.option.imagenet_pretrain:
                self.net = models.resnet18(pretrained=True)
                self.net.fc = torch.nn.Linear(512, self.option.n_class)
            else:
                self.net = models.resnet18(pretrained=False, num_classes=self.option.n_class)


        if self.option.ubnet:
            if self.option.model == 'vgg11':
                self.ubnet = model_vgg11.UBNet(num_classes=self.option.n_class)
                self.loss_ubnet = nn.CrossEntropyLoss()
                self._load_model()
            elif self.option.model == 'resnet18':
                self.ubnet = model_resnet18.UBNet(num_classes=self.option.n_class, mid_ch=self.option.mid_ch)
                self.loss_ubnet = nn.CrossEntropyLoss()
                self._load_model()
        else: 
            self.loss = nn.CrossEntropyLoss()

        if self.option.cuda:
            if self.option.ubnet:
                self.ubnet.cuda()
                self.loss_ubnet.cuda()
                self.net.cuda()
            else:
                self.net.cuda()
                self.loss.cuda()

            if torch.cuda.device_count() > 1:
                self.logger.info(f"Multi GPU : {torch.cuda.device_count()}")
                self.net = torch.nn.DataParallel(self.net)
                if self.option.ubnet:
                    self.ubnet = torch.nn.DataParallel(self.ubnet)

    def _set_optimizer(self):
        if self.option.ubnet:
            if self.option.optim == 'AdamP':
                self.optim_orth = AdamP(filter(lambda p: p.requires_grad, self.ubnet.parameters()), lr=self.option.lr,
                                        betas=(0.9, 0.999), weight_decay=self.option.weight_decay)
            elif self.option.optim == 'SGDP':
                self.optim_ort = SGDP(filter(lambda p: p.requires_grad, self.ubnet.parameters()), lr=self.option.lr,
                                      weight_decay=self.option.weight_decay, momentum=0.9, nesterov=True)
            elif self.option.optim == 'SGD':
                self.optim_orth = optim.SGD(filter(lambda p: p.requires_grad, self.ubnet.parameters()), lr=self.option.lr,
                                            weight_decay=self.option.weight_decay, momentum=0.9, nesterov=True)
            elif self.option.optim == 'Adam':
                self.optim_orth = optim.Adam(filter(lambda p: p.requires_grad, self.ubnet.parameters()), lr=self.option.lr, weight_decay=self.option.weight_decay)
            else:
                self.logger.error("wrong optimizer")
        else:
            self.optim = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.option.lr, weight_decay=self.option.weight_decay)

        lr_lambda = lambda step: self.option.lr_decay_rate ** (step // self.option.lr_decay_period)

        if self.option.ubnet:
            if self.option.lr_scheduler == 'step':
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optim_orth, lr_lambda=lr_lambda, last_epoch=-1)
            elif self.option.lr_scheduler == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim_orth, T_max=self.option.max_step, eta_min=0)
        else:
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda, last_epoch=-1)

    @staticmethod
    def _weights_init_xavier(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.xavier_normal_(m.weight.data, gain=1.0)
        elif classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight.data, gain=1.0)

    def _initialization(self):
        if self.option.is_train:
            if self.option.ubnet:
                self.ubnet.apply(self._weights_init_xavier)

            if self.option.use_pretrain:
                if self.option.checkpoint is not None:
                    self._load_model()
                else:
                    self.logger.error("no pre-trained model")

    def _mode_setting(self, is_train=True):

        if is_train:
            if self.option.ubnet:
                self.ubnet.train()
                self.net.train()
            else: self.net.train()

        else:
            if self.option.ubnet:
                self.ubnet.eval()
                self.net.eval()
            else: self.net.eval()

    def extract_features(self, images):
        if self.option.model == 'vgg11':
            """
            Orthogonal for VGG11
            """
            if torch.cuda.device_count() > 1:
                new_classifier = nn.Sequential(*list(self.net.module.children())[0])
            else:
                new_classifier = nn.Sequential(*list(self.net.children())[0])
            extractor_1 = nn.Sequential(*list(new_classifier.children())[:3]).cuda()
            extractor_2 = nn.Sequential(*list(new_classifier.children())[:6]).cuda()
            extractor_3 = nn.Sequential(*list(new_classifier.children())[:11]).cuda()
            extractor_4 = nn.Sequential(*list(new_classifier.children())[:16]).cuda()
            extractor_5 = nn.Sequential(*list(new_classifier.children())[:21]).cuda()

        elif self.option.model == 'resnet18':
            """
            Orthogonal for resnet18
            """
            if torch.cuda.device_count() > 1:
                extractor_1 = nn.Sequential(*list(self.net.module.children())[:4]).cuda()
                extractor_2 = nn.Sequential(*list(self.net.module.children())[:5]).cuda()
                extractor_3 = nn.Sequential(*list(self.net.module.children())[:6]).cuda()
                extractor_4 = nn.Sequential(*list(self.net.module.children())[:7]).cuda()
                extractor_5 = nn.Sequential(*list(self.net.module.children())[:8]).cuda()
            else:
                extractor_1 = nn.Sequential(*list(self.net.children())[:4]).cuda()
                extractor_2 = nn.Sequential(*list(self.net.children())[:5]).cuda()
                extractor_3 = nn.Sequential(*list(self.net.children())[:6]).cuda()
                extractor_4 = nn.Sequential(*list(self.net.children())[:7]).cuda()
                extractor_5 = nn.Sequential(*list(self.net.children())[:8]).cuda()

        for param in extractor_1.parameters():
            param.requires_grad = False
        for param in extractor_2.parameters():
            param.requires_grad = False
        for param in extractor_3.parameters():
            param.requires_grad = False

        feature_1 = extractor_1.forward(images)
        feature_2 = extractor_2.forward(images)
        feature_3 = extractor_3.forward(images)
        feature_4 = extractor_4.forward(images)
        feature_5 = extractor_5.forward(images)

        out = {}
        out['out1'] = feature_1
        out['out2'] = feature_2
        out['out3'] = feature_3
        out['out4'] = feature_4
        out['out5'] = feature_5

        return out

    def _train_step(self, data_loader, step):
        self._mode_setting(is_train=True)
        
        loss_sum = 0.
        loss_orth_sum = 0.
        loss_conv_sum = 0.
        loss_trans_sum = 0.
        total_num_correct = 0
        total_num_train = 0
        t_bar = tqdm(data_loader, leave=False)
        for i, cur_data in enumerate(t_bar):
            if self.option.data == 'imagenet':
                images, labels, bias_labels = cur_data
            else:
                images, labels = cur_data
            
            images = self._get_variable(images)
            labels = self._get_variable(labels)
            total_num_train += images.shape[0]


            if self.option.ubnet:
                out = self.extract_features(images)

                self.optim_orth.zero_grad()
                pred_label_orth, loss_conv, loss_trans = self.ubnet(out)
                loss_orth = self.loss_ubnet(pred_label_orth, torch.squeeze(labels))
                loss_orth_sum += loss_orth
                loss_conv_sum += loss_conv
                loss_trans_sum += loss_trans
                loss_orth.backward()
                self.optim_orth.step()

                total_num_correct += self._num_correct(pred_label_orth, labels, topk=1).data
                t_bar.set_description(f"Train Loss {round(float(loss_orth)/images.shape[0], 6)}")
                    
            else:
                """
                Not ubnet training
                """
                self.optim.zero_grad()
                pred_label = self.net(images)
                loss = self.loss(pred_label, torch.squeeze(labels))
                loss_sum += loss
                loss.backward()
                self.optim.step()
            self.writer.add_scalar('Loss/train_step', loss_orth/images.shape[0], step * len(data_loader) + i)
            self.writer.add_scalar('Accuracy/train_step',
                                    self._num_correct(pred_label_orth, labels, topk=1).data / images.shape[0],
                                    step * len(data_loader) + i)                                     

        if self.option.ubnet:
            msg = f"[TRAIN] ORTH LOSS : {round(loss_orth_sum.cpu().detach().numpy()/total_num_train,6)} " \
                  f"ACCURACY {round(float(total_num_correct) / total_num_train, 6)} " \
                  f"lr: {self.optim_orth.param_groups[0]['lr']}"
        else:
            msg = f"[TRAIN] BASE LOSS : {loss_sum/len(data_loader)}"
        self.logger.info(msg)

        self.writer.add_scalars('Loss/epoch', {'train': loss_orth_sum/ len(data_loader)}, step)
        self.writer.add_scalars('Accuracy/epoch', {'train': float(total_num_correct)/total_num_train}, step)
        self.writer.add_scalar('Learning rate', self.optim_orth.param_groups[0]['lr'], step)


    def _validate(self, data_loader, step, valid_type=None):
        self._mode_setting(is_train=False)

        if not self.option.is_train:
            self.logger.info("not in training process")
            self._initialization()
            if self.option.checkpoint_orth is not None:
                self._load_model()
            else:
                self.logger.error("No trained model")
                sys.exit()

        total_num_correct = 0.
        total_num_correct_orth = 0.
        total_num_test = 0.
        total_loss = 0.
        total_loss_orth = 0.
        total_loss_conv = 0.
        total_loss_trans = 0.

        for i, (images,labels) in enumerate(tqdm(data_loader, leave=False)):
            
            images = self._get_variable(images)
            labels = self._get_variable(labels)
            
            batch_size = images.shape[0]
            total_num_test += batch_size
            if self.option.ubnet:
                self.optim_orth.zero_grad()
                out = self.extract_features(images)

                pred_label_orth, loss_conv, loss_trans = self.ubnet(out)
                loss_orth = self.loss_ubnet(pred_label_orth, torch.squeeze(labels))
                total_num_correct_orth += self._num_correct(pred_label_orth,labels,topk=1).data
                total_loss_orth += loss_orth.data*batch_size
                total_loss_conv += loss_conv
                total_loss_trans += loss_trans

            if not self.option.ubnet:
                self.optim.zero_grad()
                pred_label = self.net(images)
                loss = self.loss(pred_label, torch.squeeze(labels))
                
                total_num_correct += self._num_correct(pred_label,labels,topk=1).data
                total_loss += loss.data*batch_size

        if self.option.ubnet:
            avg_loss_orth = total_loss_orth/total_num_test
            avg_acc_orth = total_num_correct_orth/total_num_test
            msg = f"[EVALUATION({valid_type})] (step {step}) LOSS : {np.round(avg_loss_orth.cpu().detach().numpy(),6)}, " \
                  f"ACCURACY : {np.round(avg_acc_orth.cpu().detach().numpy(),6)} "
            self.writer.add_scalars('Loss/epoch', {f'valid_{valid_type}': avg_loss_orth}, step)
            self.writer.add_scalars('Accuracy/epoch', {f'valid_{valid_type}': avg_acc_orth}, step)
            self.logger.info(msg)

        else:

            avg_loss = total_loss/total_num_test
            avg_acc = float(total_num_correct)/total_num_test
            if valid_type != None:
                msg = f"[EVALUATION - {valid_type}] LOSS : {avg_loss}, ACCURACY : {avg_acc}"
            else:
                msg = f"[EVALUATION] LOSS : {avg_loss}, ACCURACY : {avg_acc}"
            self.logger.info(msg)
        
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
        n_correct = (predicted == labels).sum().item()
        return n_correct

    def _validate_imagenet(self, data_loader, step=0, valid_type='',
                        num_clusters=9,
                        num_cluster_repeat=3):
        self._mode_setting(is_train=False)

        if not self.option.is_train:
            print("not in training process")
            self._initialization()
            if self.option.checkpoint_orth is not None:
                self._load_model()
            else:
                print("No trained model")
                sys.exit()

        total_num_correct_orth = 0.
        total_num_test = 0.
        total_loss_orth = 0.
        total_loss_conv = 0.
        total_loss_trans = 0.

        total = 0
        f_correct = 0
        num_correct = [np.zeros([self.option.n_class, num_clusters]) for _ in range(num_cluster_repeat)]
        num_instance = [np.zeros([self.option.n_class, num_clusters]) for _ in range(num_cluster_repeat)]

        for i, (images, labels, bias_labels) in enumerate(tqdm(data_loader)):

            images = self._get_variable(images)
            labels = self._get_variable(labels)

            batch_size = images.shape[0]
            total_num_test += batch_size
            total += batch_size
            if self.option.ubnet:
                self.optim_orth.zero_grad()
                out = self.extract_features(images)

                pred_label_orth, loss_conv, loss_trans = self.ubnet(out)

                if valid_type == 'unbiased':
                    num_correct, num_instance = self.imagenet_unbiased_accuracy(pred_label_orth.data, labels,
                                                                                bias_labels,
                                                                                num_correct, num_instance,
                                                                                num_cluster_repeat)
                else:
                    f_correct += self.n_correct(pred_label_orth, labels)

                loss_orth = self.loss_ubnet(pred_label_orth, torch.squeeze(labels))
                total_num_correct_orth += self._num_correct(pred_label_orth, labels, topk=1).data
                total_loss_orth += loss_orth.data * batch_size
                total_loss_conv += loss_conv
                total_loss_trans += loss_trans

        if self.option.ubnet:
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

                avg_acc_orth = f_correct / num_cluster_repeat
            else:
                avg_acc_orth = f_correct / total

            avg_loss_orth = total_loss_orth / total_num_test
            msg = f"[EVALUATION({valid_type})] step{step} LOSS : {avg_loss_orth}, ACCURACY : {avg_acc_orth}"
            self.writer.add_scalars('Loss/epoch', {f'valid_{valid_type}': avg_loss_orth}, step)
            self.writer.add_scalars('Accuracy/epoch', {f'valid_{valid_type}': avg_acc_orth}, step)
            self.cur_acc_orth = avg_acc_orth
        self.logger.info(msg)

    def _num_correct(self,outputs,labels,topk=1):
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
        if self.option.ubnet:
            if torch.cuda.device_count() > 1:
                torch.save({
                    'step': step,
                    'optim_state_dict': self.optim_orth.state_dict(),
                    'net_state_dict': self.ubnet.module.state_dict(),
                    'feature_state_dict': self.net.module.state_dict(),
                }, os.path.join(self.option.save_dir, self.option.exp_name, f'checkpoint_step_{step}.pth'))
            else:
                torch.save({
                    'step': step,
                    'optim_state_dict': self.optim_orth.state_dict(),
                    'net_state_dict': self.ubnet.state_dict(),
                    'feature_state_dict': self.net.state_dict(),
                }, os.path.join(self.option.save_dir,self.option.exp_name, f'checkpoint_step_{step}.pth'))
        else:
            torch.save({
                'step': step,
                'optim_state_dict': self.optim.state_dict(),
                'net_state_dict': self.net.state_dict()
            }, os.path.join(self.option.save_dir,self.option.exp_name, f'checkpoint_step_{step}.pth'))

        print('checkpoint saved. step : %d'%step)


    def _load_model(self):
        if self.option.ubnet:
            if self.option.checkpoint_orth:
                self.logger.info(f"load from checkpoint: {self.option.checkpoint_orth}")
                orth_ckpt = torch.load(self.option.checkpoint_orth)
                if 'feature_state_dict' in orth_ckpt.keys():
                    self.ubnet.load_state_dict(orth_ckpt['net_state_dict'], strict=True)
                    self.net.load_state_dict(orth_ckpt['feature_state_dict'], strict=True)
                else:
                    self.ubnet.load_state_dict(orth_ckpt['net_state_dict'], strict=True)
                    ckpt = torch.load(self.option.checkpoint)
                    self.net.load_state_dict(ckpt['net_state_dict'], strict=True)
            elif self.option.checkpoint:
                ckpt = torch.load(self.option.checkpoint)
                self.net.load_state_dict(ckpt['net_state_dict'], strict=True)
        elif self.option.checkpoint:
            ckpt = torch.load(self.option.checkpoint)
            self.net.load_state_dict(ckpt['net_state_dict'], strict=True)

    def train(self, train_loader, val_loader=None, val_loader_bias=None, val_loaders=None, val_types=None):
        self._initialization()
        if self.option.checkpoint is not None:
            self._load_model()

        self._mode_setting(is_train=True)
        start_epoch = 0
        for step in range(start_epoch, self.option.max_step):
            self._train_step(train_loader,step)
            self.scheduler.step()

            if step == 1 or step % self.option.save_step == 0 or step == (self.option.max_step-1):
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
