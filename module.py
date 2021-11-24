import torch
import torch.nn as nn
import torch.optim as optim


class Orthorconv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride=1, padding=0, bias=True, groups=1):
        super(Orthorconv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.groups = groups
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding, bias=bias, groups=groups)
        self.opt_orth = optim.Adam(self.parameters(), lr=1e-3, betas=(0.5, 0.99))
        self.out_channel = out_channel
        self.in_channel = in_channel

    def orthogonal_update(self):
        self.zero_grad()
        self.loss = cos_similarity(self.conv.weight.view(self.groups, -1))
        self.loss.backward()
        self.opt_orth.step()

    def forward(self, feat):
        if self.training:
            self.orthogonal_update()
        else:
            self.loss = 0.
        return self.conv(feat), self.loss


class OrthorTransform(nn.Module):
    def __init__(self, c_dim, feat_hw, groups): #feat_hw: width or height (let width == heigt)
        super(OrthorTransform, self).__init__()

        self.groups = groups
        self.c_dim = c_dim
        self.feat_hw = feat_hw
        self.weight = nn.Parameter(torch.randn(1, c_dim, feat_hw, feat_hw))
        self.opt_orth = optim.Adam(self.parameters(), lr=1e-3, betas=(0.5, 0.99))

    def orthogonal_update(self):
        self.zero_grad()
        self.loss = cos_similarity(self.weight.view(self.groups, -1))
        self.loss.backward()
        self.opt_orth.step()

    def forward(self, feat):
        if self.training:
            self.orthogonal_update()
        else:
            self.loss = 0.
        pred = feat * self.weight.expand_as(feat)
        return pred.mean(-1).mean(-1), self.loss


# Q module that utilzes the orthogonal regularized conv and transformer layers
class CodeReduction(nn.Module):
    def __init__(self, c_dim, feat_hw, blocks = 5, prob=False):
        super(CodeReduction, self).__init__()
        if prob:
            c_dim *= 2
        
        self.main = nn.Sequential(
            Orthorconv2d(c_dim, c_dim, 3, 1, 1, bias=True, groups=blocks)
        )

        self.trans = OrthorTransform(c_dim=c_dim, feat_hw=feat_hw, groups = blocks)
    
    def forward(self, feat):
        feat,loss_conv = self.main(feat)
        pred_c,loss_trans = self.trans(feat)
        return pred_c.view(feat.size(0), -1), loss_conv, loss_trans

def cos_similarity(weight):
    weight = weight / weight.norm(dim=-1).unsqueeze(-1)
    cos_distance = torch.mm(weight, weight.transpose(1,0))

    #diagonal element -> 0
    cosine_matrix = cos_distance.pow(2)
    cosine_matrix.fill_diagonal_(0)

    return cosine_matrix.mean()
