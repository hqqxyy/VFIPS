import tqdm
import torch
import networks
import loss
import random
import numpy as np
import data.dataset
from utils import *
from option import opt


# fix sed
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    mkdirifnotexist(opt.expdir)
    # object setting
    if opt.model == 'multiscale_v33':
        print('=> multiscale v33')
        objectSettings = {
            'intEpochs': 20,
            'intBatchsize': 8,
        }
        ckptpath = None
        moduleNetwork = networks.get_model('multiscale_v33', depth_ksize=opt.depth_ksize)
        moduleLoss = loss.RealBCELoss().cuda()
        objectOptimizer = torch.optim.AdamW(list(moduleNetwork.parameters()) + list(moduleLoss.parameters()), lr=opt.lr, betas=(0.5, 0.999))
        objectScheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=objectOptimizer, milestones=[30, 40], gamma=0.5)
        # objectScheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=objectOptimizer, milestones=[30, 40], gamma=0.5)

        def forward(tenRef, tenFirst, tenSecond, tenJudge, tenFlowFirst, tenFlowSecond, tenFlowGt):
            candidate_size = range(192, 256)
            video_size = random.choice(candidate_size)

            N, B, C, H, W = tenRef.size()
            tenRef = tenRef.view(-1, C, H, W).clone()
            tenRef = torch.nn.functional.interpolate(tenRef, size=video_size, mode='bilinear', align_corners=False)
            tenRef = tenRef.view(N, B, C, video_size, video_size).clone()

            tenFirst = tenFirst.view(-1, C, H, W).clone()
            tenFirst = torch.nn.functional.interpolate(tenFirst, size=video_size, mode='bilinear', align_corners=False)
            tenFirst = tenFirst.view(N, B, C, video_size, video_size).clone()

            tenSecond = tenSecond.view(-1, C, H, W).clone()
            tenSecond = torch.nn.functional.interpolate(tenSecond, size=video_size, mode='bilinear', align_corners=False)
            tenSecond = tenSecond.view(N, B, C, video_size, video_size).clone()

            tenDisFirst = moduleNetwork(tenRef, tenFirst)
            tenDisSecond = moduleNetwork(tenRef, tenSecond)
            tenLoss = torch.mean(moduleLoss(tenDisFirst, tenDisSecond, tenJudge))
            return tenLoss

        def step():
            torch.nn.utils.clip_grad_norm_(moduleNetwork.parameters(), 0.1)

    else:
        raise NotImplementedError


    # 90% human label + all autodata for training
    objectTrain = torch.utils.data.DataLoader(
        batch_size=objectSettings['intBatchsize'],
        shuffle=True,
        num_workers=12,
        drop_last=True,
        dataset=data.dataset.PVIDEODatasetTrainV2(
            dataroot=get_dataset_dir(),
            autodata=True,
            boolflow=False
        )
    )

    # 10% human label for validation
    objectTest = torch.utils.data.DataLoader(
        batch_size=objectSettings['intBatchsize'],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        dataset=data.dataset.PVIDEODatasetTestV2Human(
            dataroot=get_dataset_dir(),
            boolflow=False
        )
    )

    for intEpoch in range(objectSettings['intEpochs']):
        torch.set_grad_enabled(True)
        moduleNetwork.train()
        moduleLoss.train()

        for tenRef, tenFirst, tenSecond, tenJudge, tenFlowFirst, tenFlowSecond, tenFlowGt in tqdm.tqdm(objectTrain):
            objectOptimizer.zero_grad()
            forward(tenRef.cuda(), tenFirst.cuda(), tenSecond.cuda(), tenJudge.cuda(), tenFlowFirst.cuda(), tenFlowSecond.cuda(), tenFlowGt.cuda()).backward()
            step()
            objectOptimizer.step()

        objectScheduler.step()
        torch.set_grad_enabled(False)
        moduleNetwork.eval()
        moduleLoss.eval()
        dblTest = []
        for tenRef, tenFirst, tenSecond, tenJudge, tenFlowFirst, tenFlowSecond, tenFlowGt in tqdm.tqdm(objectTest):
            dblTest.append(forward(tenRef.cuda(), tenFirst.cuda(), tenSecond.cuda(), tenJudge.cuda(), tenFlowFirst.cuda(), tenFlowSecond.cuda(), tenFlowGt.cuda()).item())

        print('test ', intEpoch, ': ', np.mean(dblTest))
        with open(opt.expdir + 'train.log', 'a+') as objectFile:
            objectFile.write(str(np.mean(dblTest)) + '\n')

        torch.save(moduleNetwork.state_dict(), opt.expdir + 'model.pytorch')
