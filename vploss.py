import torch

class ScalingLayer(torch.nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(torch.nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [torch.nn.Dropout(),] if(use_dropout) else []
        layers += [torch.nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)


def upsample(in_tens, out_HW=(64,64)): # assumes scale factor is same for H and W
    return torch.nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)


class CALayer(torch.nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = torch.nn.Sequential(
                torch.nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                torch.nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class Extractor(torch.nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()

        self.moduleFirst = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSecond = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleThird = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFourth = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFifth = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSixth = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )


    def forward(self, tensorInput):
        tensorFirst = self.moduleFirst(tensorInput)
        tensorSecond = self.moduleSecond(tensorFirst)
        tensorThird = self.moduleThird(tensorSecond)
        tensorFourth = self.moduleFourth(tensorThird)
        tensorFifth = self.moduleFifth(tensorFourth)
        tensorSixth = self.moduleSixth(tensorFifth)

        return [tensorFirst, tensorSecond, tensorThird, tensorFourth, tensorFifth, tensorSixth]

# from V10, add the source features
# 0.847059
class MultiScaleV11(torch.nn.Module):
    def __init__(self):
        super(MultiScaleV11, self).__init__()
        self.moduleExtractor = Extractor()
        chns = [16, 32, 64, 96, 128, 192]
        out_chn = 32
        merge_layers = []
        for chn in chns:
            merge_layers.append(
                torch.nn.Sequential(
                    CALayer(12 * chn * 3),
                    torch.nn.Conv2d(in_channels=12 * chn * 3, out_channels=out_chn, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=out_chn, out_channels=out_chn, kernel_size=3, stride=1, padding=1),
                    CALayer(out_chn),
                )
            )
        self.merge_layers = torch.nn.ModuleList(merge_layers)

    def forward(self, inputFirst, inputSecond):
        B, V, C, H, W = inputFirst.size()
        inputFirst = inputFirst.view(B * V, C, H, W)
        inputSecond = inputSecond.view(B * V, C, H, W)
        tensorFeasFirst = self.moduleExtractor(inputFirst)
        tensorFeasSecond = self.moduleExtractor(inputSecond)

        tensorFeas = []
        for tensorFeaFirst, tensorFeaSecond in zip(tensorFeasFirst, tensorFeasSecond):
            tensorFeaFirst = normalize_tensor(tensorFeaFirst)
            tensorFeaSecond = normalize_tensor(tensorFeaSecond)
            tensorFea = torch.abs(tensorFeaFirst - tensorFeaSecond)
            tensorFea = torch.cat([tensorFeaFirst, tensorFeaSecond, tensorFea], dim=1)
            tensorFeas.append(tensorFea)

        tensorFeas = [p.view(B, -1, p.size(2), p.size(3)) for p in tensorFeas]
        tensorOuts = []
        for merge_layer, tensorFea in zip(self.merge_layers, tensorFeas):
            tensorOut = merge_layer(tensorFea)
            tensorOuts.append(tensorOut)
        return tensorOuts


# from LPIPS 3D, we first calculate the diff, then we get the final result
class LPIPS_3D_Diff(torch.nn.Module):
    def __init__(self,
                 net='alex_3d',
                 lpips_method='lpips',
                 ckpt_path=''
                 ):

        super(LPIPS_3D_Diff, self).__init__()

        self.pnet_type = net
        self.spatial = False
        self.lpips_method = lpips_method
        self.scaling_layer = ScalingLayer()

        net_type = MultiScaleV11
        self.net = net_type()
        self.chns = [32, 32, 32, 32, 32, 32]

        self.L = len(self.chns)

        if self.lpips_method == 'lpips':
            use_dropout = True
            lins = []
            for chn in self.chns:
                lins.append(NetLinLayer(chn, use_dropout=use_dropout))
            self.lins = torch.nn.ModuleList(lins)
        elif self.lpips_method == 'at1':
            lins = []
            for chn in self.chns:
                lins.append(torch.nn.Sequential(
                    torch.nn.Dropout(0.2),
                    torch.nn.Conv2d(chn, 32, 1, stride=1, padding=0, bias=False),
                    torch.nn.ReLU()
                ))
            self.lins = torch.nn.ModuleList(lins)
            self.lins2 = torch.nn.Sequential(
                torch.nn.Dropout(0.2),
                torch.nn.Conv2d(32 * len(self.chns), 1, 1, stride=1, padding=0, bias=False),
            )

        self.load_state_dict(torch.load('exp/multiscale_family_multiscale_v11/model.pytorch'))


    def forward(self, in0, in1, normalize=False):
        if normalize:
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)
        diffs = self.net.forward(in0_input, in1_input)

        if self.lpips_method == 'lpips':
            if (self.spatial):
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
            val = res[0]
            for l in range(1, self.L):
                val += res[l]
        elif self.lpips_method == 'at1':
            res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
            res = torch.cat(res, dim=1)
            val = self.lins2(res)
        else:
            if (self.spatial):
                res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]

            val = res[0]
            for l in range(1, self.L):
                val += res[l]

        return val


if __name__ == '__main__':
    a = LPIPS_3D_Diff()
    a.load_state_dict(torch.load('exp/multiscale_family_multiscale_v11/model.pytorch'))


