import torch
import torch.nn as nn


class Dist2LogitLayer(nn.Module):
    ''' takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) '''
    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        if(use_sigmoid):
            layers += [nn.Sigmoid(),]
        self.model = nn.Sequential(*layers)

    def forward(self,d0,d1,eps=0.1):
        return self.model.forward(torch.cat((d0,d1,d0-d1,d0/(d1+eps),d1/(d0+eps)),dim=1))


class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        # self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        # per = (judge+1.)/2.
        self.logit = self.net.forward(d0,d1)
        return self.loss(self.logit, judge)


class RealBCELoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(RealBCELoss, self).__init__()
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        logit = torch.nn.functional.sigmoid(d0 - d1)
        if len(logit.size()) == 4:
            assert logit.size(2) == 1
            assert logit.size(3) == 1
            logit = logit[:, :, 0, 0]
        return self.loss(logit, judge)


class RealL1Loss(nn.Module):
    def __init__(self, reduction='sum'):
        super(RealL1Loss, self).__init__()
        self.loss = torch.nn.L1Loss(reduction=reduction)

    def forward(self, d0, d1, judge):
        logit = torch.nn.functional.sigmoid(d0 - d1)
        if len(logit.size()) == 4:
            assert logit.size(2) == 1
            assert logit.size(3) == 1
            logit = logit[:, :, 0, 0]
        return self.loss(logit, judge)


class RealBCEWithLogitsLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(RealBCEWithLogitsLoss, self).__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, d0, d1, judge):
        logit = d0 - d1
        if len(logit.size()) == 4:
            assert logit.size(2) == 1
            assert logit.size(3) == 1
            logit = logit[:, :, 0, 0]
        return self.loss(logit, judge)


