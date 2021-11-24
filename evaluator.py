"""ReBias
We referenced https://github.com/clovaai/rebias for 9-Class imagenet experiment
"""
import torch
import numpy as np

def n_correct(pred, labels):
    _, predicted = torch.max(pred.data, 1)
    n_correct = (predicted == labels).sum().item()
    return n_correct


class EvaluatorBase(object):
    def __init__(self, device='cuda'):
        self.device = device

    @torch.no_grad()
    def evaluate_acc(self, dataloader, model):
        model.eval()

        total = 0
        correct = 0

        for x, labels, index in dataloader:
            x = x.to(self.device)
            labels = labels.to(self.device)
            pred = model(x, logits_only=True)

            batch_size = labels.size(0)
            total += batch_size
            correct += n_correct(pred, labels)

        return correct / total

    @torch.no_grad()
    def evaluate_rebias(self, dataloader, rebias_model,
                        outer_criterion=None,
                        inner_criterion=None,
                        **kwargs):
        raise NotImplementedError

class ImageNetEvaluator(EvaluatorBase):
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

    @torch.no_grad()
    def evaluate_rebias(self, dataloader, rebias_model,
                        outer_criterion=None,
                        inner_criterion=None,
                        num_classes=9,
                        num_clusters=9,
                        num_cluster_repeat=3,
                        key=None):
        rebias_model.eval()

        total = 0
        f_correct = 0
        num_correct = [np.zeros([num_classes, num_clusters]) for _ in range(num_cluster_repeat)]
        num_instance = [np.zeros([num_classes, num_clusters]) for _ in range(num_cluster_repeat)]
        g_corrects = [0 for _ in rebias_model.g_nets]

        if outer_criterion.__class__.__name__ in ['LearnedMixin', 'RUBi']:
            """For computing HSIC loss only.
            """
            outer_criterion = None

        outer_loss = [0 for _ in rebias_model.g_nets]
        inner_loss = [0 for _ in rebias_model.g_nets]

        for x, labels, bias_labels in dataloader:
            x = x.to(self.device)
            labels = labels.to(self.device)
            for bias_label in bias_labels:
                bias_label.to(self.device)

            f_pred, g_preds, f_feat, g_feats = rebias_model(x)

            batch_size = labels.size(0)
            total += batch_size

            if key == 'unbiased':
                num_correct, num_instance = self.imagenet_unbiased_accuracy(f_pred.data, labels, bias_labels,
                                                                            num_correct, num_instance, num_cluster_repeat)
            else:
                f_correct += n_correct(f_pred, labels)
                for idx, g_pred in enumerate(g_preds):
                    g_corrects[idx] += n_correct(g_pred, labels)

            if outer_criterion:
                for idx, g_pred in enumerate(g_preds):
                    outer_loss[idx] += batch_size * outer_criterion(f_pred, g_pred).item()

            if inner_criterion:
                for idx, g_pred in enumerate(g_preds):
                    inner_loss[idx] += batch_size * inner_criterion(f_pred, g_pred).item()

        if key == 'unbiased':
            for k in range(num_cluster_repeat):
                x, y = [], []
                _num_correct, _num_instance = num_correct[k].flatten(), num_instance[k].flatten()
                for i in range(_num_correct.shape[0]):
                    __num_correct, __num_instance = _num_correct[i], _num_instance[i]
                    if __num_instance >= 10:
                        x.append(__num_instance)
                        y.append(__num_correct / __num_instance)
                f_correct += sum(y) / len(x)

            ret = {'f_acc': f_correct / num_cluster_repeat}
        else:
            ret = {'f_acc': f_correct / total}

        for idx, (_g_correct, _outer_loss, _inner_loss) in enumerate(zip(g_corrects, outer_loss, inner_loss)):
            ret['g_{}_acc'.format(idx)] = _g_correct / total
            ret['outer_{}_loss'.format(idx)] = _outer_loss / total
            ret['inner_{}_loss'.format(idx)] = _inner_loss / total
        return ret