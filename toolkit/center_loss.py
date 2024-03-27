
import torch
import torch.nn as nn

def Other_label(labels,num_classes):
    index=torch.randint(num_classes, (labels.shape[0],)).to(labels.device)
    other_labels=labels+index
    other_labels[other_labels >= num_classes]=other_labels[other_labels >= num_classes]-num_classes
    return other_labels


class TripCenterLoss_margin(nn.Module):

    def __init__(self, num_classes=10, feat_dim=312, use_gpu=True, pre_center=None):
        super(TripCenterLoss_margin, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        if pre_center==None:
            if self.use_gpu:
                self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())  ## nn.Parameter() 将tensor绑定到网络参数里，随之优化, 所以这个中心其实是习得的？
            else:
                self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        else:
            self.centers = pre_center
    def forward(self, x, labels,margin, incenter_weight):
        other_labels = Other_label(labels, self.num_classes)
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # distmat.addmm_(1, -2, x, self.centers.t())  ## 1*distmat - 2*x*self.ceners.t() 结合distmat的计算，其实在算x与center的欧式距离
        distmat.addmm_(x,self.centers.t(),beta=1,alpha=-2)


        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat[mask]
        other_labels = other_labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask_other = other_labels.eq(classes.expand(batch_size, self.num_classes))
        dist_other = distmat[mask_other]
        loss = torch.max(margin+incenter_weight*dist-(1-incenter_weight)*dist_other,torch.tensor(0.0).cuda()).sum() / batch_size
        return loss

