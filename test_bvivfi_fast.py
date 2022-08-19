import tqdm
import torch
import networks
import glob
import numpy as np
from utils import *
from option import opt
from PIL import Image
import cv2
import torchvision.transforms as transforms
from scipy.stats import pearsonr, spearmanr, kendalltau



def test_plcc(predpath):
    gtpath = './bvi_vfi_dmos.pkl'
    gt_dmos = load_pkl(gtpath)
    pred_dmos = load_pkl(predpath)

    local_pearson = []
    local_spearmanr = []
    local_kendalltau = []
    videolist = list(set([p[:p.rfind('_') + 1] for p in gt_dmos.keys()])) 
    for videoname in videolist:
        gt = []
        pred = []
        for method in ["Repeat","Average","DVF","QVI","STMFNet"]:
            gt.append(gt_dmos[videoname + method])
            tmp = np.array(pred_dmos[videoname + method])
            tmp = tmp[tmp != np.inf]
            tmp = np.mean(tmp)
            pred.append(tmp)

        pearson_corr, pearson_p = pearsonr(gt, pred)
        local_pearson.append(pearson_corr)

        spearmanr_corr, spearmanr_p = spearmanr(gt, pred)
        local_spearmanr.append(spearmanr_corr)

        kendalltau_corr, kendalltau_p = kendalltau(gt, pred)
        local_kendalltau.append(kendalltau_corr)


    print('*' * 50)
    print(predpath)

    print('plcc', np.mean(local_pearson))
    print('srcc', np.mean(local_spearmanr))
    print('krcc', np.mean(local_kendalltau))



if __name__ == '__main__':
    # object setting
    print('=> model %s'%opt.model)
    moduleNetwork = networks.get_model(opt.model, depth_ksize=opt.depth_ksize,opt=opt)
    print('=> load model from ckpt', opt.expdir)
    moduleNetwork.load_state_dict(torch.load(opt.expdir + 'model.pytorch'))

    def estimate(tenRef, tenVideo):
        tenDis = moduleNetwork(tenRef, tenVideo)
        return tenDis

    torch.set_grad_enabled(False)
    moduleNetwork.eval()


    dataroot = get_dataset_dir('bvivfi')
    dmos = load_pkl(dataroot + 'bvi_vfi_dmos.pkl')
    gtlist = glob.glob(dataroot + 'videos/*/*_GT.mp4')
    gtlist.sort()

    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    transform = transforms.Compose(transform_list)

    scores = {}
    for gtpath in tqdm.tqdm(gtlist):
        for method in ['Average', 'DVF', 'QVI', 'Repeat', 'STMFNet']:
            dstpath = gtpath.replace('_GT.mp4', '_' + method + '.mp4')
            dstkey = dstpath.split('/')[-1][:-4]

            cap_gt = cv2.VideoCapture(gtpath)
            if not cap_gt.isOpened():
                print("Error opening GT video")
            frame_count = int(cap_gt.get(cv2.CAP_PROP_FRAME_COUNT))

            cap_video = cv2.VideoCapture(dstpath)
            if not cap_video.isOpened():
                print("Error opening Dis video")
 
            # stride 12
            # for start_id in range(0, frame_count - 12):
            for start_id in range(0, frame_count - 12, 12):
                video = []
                gt = []
                for i in range(12):
                    ret, videoimg = cap_video.read()
                    assert ret
                    videoimg = cv2.cvtColor(videoimg, cv2.COLOR_BGR2RGB)
                    videoimg = Image.fromarray(videoimg)
                    videoimg = transform(videoimg).unsqueeze(0)

                    ret, gtimg = cap_gt.read()
                    assert ret
                    gtimg = cv2.cvtColor(gtimg, cv2.COLOR_BGR2RGB)
                    gtimg = Image.fromarray(gtimg)
                    gtimg = transform(gtimg).unsqueeze(0)

                    video.append(videoimg)
                    gt.append(gtimg)
                
                video = torch.cat(video, dim=0)
                gt = torch.cat(gt, dim=0)

                video = video.unsqueeze(0).cuda()
                gt = gt.unsqueeze(0).cuda()

                dis = estimate(gt.cuda(), video.cuda())
                dis = dis.data.cpu().numpy().flatten()

                scores.setdefault(dstkey, []).append(dis)

   
    logpath = '%s/test_dmos.pkl'%(opt.expdir)
    save_pkl(logpath, scores)

    test_plcc(logpath)


