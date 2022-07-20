import os
import random
import torch
import pickle
import numpy as np
import glob
import cv2

from PIL import Image
import torchvision.transforms as transforms


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class PVIDEODataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, boolflow=False, booltrain=False, boolVideoDir=False, boolBinary=True):
        self.dataroot = dataroot
        self.boolflow = boolflow
        self.booltrain = booltrain
        self.boolVideoDir = boolVideoDir
        self.boolBinary = boolBinary

        # following lpips, we do the normalization
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        videodir = self.videolist[index]
        videodir, label = videodir.split(' ')

        video1 = []
        video2 = []
        gt = []
        for i in range(12):
            v1path = '%s/%s/video1_%d.png' % (self.dataroot, videodir, i)
            v2path = '%s/%s/video2_%d.png' % (self.dataroot, videodir, i)
            gtpath = '%s/%s/gt_%d.png' % (self.dataroot, videodir, i)

            v1img = Image.open(v1path).convert('RGB')
            v1img = self.transform(v1img).unsqueeze(0)

            v2img = Image.open(v2path).convert('RGB')
            v2img = self.transform(v2img).unsqueeze(0)

            gtimg = Image.open(gtpath).convert('RGB')
            gtimg = self.transform(gtimg).unsqueeze(0)

            video1.append(v1img)
            video2.append(v2img)
            gt.append(gtimg)

        video1 = torch.cat(video1, dim=0)
        video2 = torch.cat(video2, dim=0)
        gt = torch.cat(gt, dim=0)

        # judge_img = np.array(float(label)).reshape((1, 1, 1,))
        
        if self.boolBinary:
            label = np.float(float(label) > 0.5)
        judge_img = np.array(float(label)).reshape((1,))
        judge_img = torch.FloatTensor(judge_img)

        if not self.boolflow:
            dummyflow = np.array(-1).reshape((1, 1, 1,))
            dummyflow = torch.FloatTensor(dummyflow)
            flow1 = dummyflow
            flow2 = dummyflow
            flowgt = dummyflow
        else:
            flow1 = []
            flow2 = []
            flowgt = []
            for i in range(11):
                v1path = '%s/%s/video1_%d-%d.pkl' % (self.dataroot, videodir, i, i + 1)
                v2path = '%s/%s/video2_%d-%d.pkl' % (self.dataroot, videodir, i, i + 1)
                gtpath = '%s/%s/gt_%d-%d.pkl' % (self.dataroot, videodir, i, i + 1)
                flow1.append(load_pkl(v1path).transpose(2, 0, 1)[np.newaxis, :])
                flow2.append(load_pkl(v2path).transpose(2, 0, 1)[np.newaxis, :])
                flowgt.append(load_pkl(gtpath).transpose(2, 0, 1)[np.newaxis, :])

            # norm by the height, width 256
            flow1 = np.concatenate(flow1, axis=0) / 256.0
            flow2 = np.concatenate(flow2, axis=0) / 256.0
            flowgt = np.concatenate(flowgt, axis=0) / 256.0

        if self.booltrain and random.random() < 0.5:
            video1, video2 = video2, video1
            flow1, flow2 = flow2, flow1
            judge_img = 1 - judge_img

        if self.boolVideoDir:
            return gt, video1, video2, judge_img, flow1, flow2, flowgt, videodir
        else:
            return gt, video1, video2, judge_img, flow1, flow2, flowgt

    def __len__(self):
        return len(self.videolist)


class PVIDEODatasetTrainV2(PVIDEODataset):
    def __init__(self, dataroot, autodata=True, boolflow=False):
        super(PVIDEODatasetTrainV2, self).__init__(dataroot, boolflow, True, boolBinary=True)

        videolist = []
        with open(self.dataroot + 'eccv_human_train.txt', 'r') as f:
            videolist += f.read().splitlines()

        if autodata:
            with open(self.dataroot + 'eccv_auto_train.txt', 'r') as f:
                videolist += f.read().splitlines()
        else:
            print('=> Warning: No auto data for training')

        random.shuffle(videolist)
        self.videolist = videolist
        print('=> load %d samples from %s' % (len(self.videolist), dataroot))


class PVIDEODatasetTestV2Human(PVIDEODataset):
    def __init__(self, dataroot, boolflow=False, boolvideodir=False):
        super(PVIDEODatasetTestV2Human, self).__init__(dataroot, boolflow, False, boolvideodir, boolBinary=False)
        with open(self.dataroot + 'eccv_human_test.txt', 'r') as f:
            videolist = f.read().splitlines()
        self.videolist = videolist
        self.videolist.sort()
        print('=> load %d samples from %s' % (len(self.videolist), dataroot))


