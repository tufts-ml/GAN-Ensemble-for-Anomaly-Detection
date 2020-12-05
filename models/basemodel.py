import os
import torch
import pandas as pd
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.evaluate import roc
from collections import OrderedDict
from utils import Visualizer
from dataloader.dataloader import load_data
from models.networks import define_D, define_G, MLP_Discriminator, MLP_Decoder, MLP_Encoder
from utils.loss import lat_loss, con_loss,l2_loss
from utils import weights_init

torch.autograd.set_detect_anomaly(True)


class ANBase:
    def __init__(self, opt):
        self.opt = opt
        if self.opt.batchsize % self.opt.split != 0:
            raise ValueError("#batchsize must be divisible w.r.t #split ")
        self.opt.batchsize //= self.opt.split
        self.epoch = self.opt.niter
        self.visualizer = Visualizer(self.opt)
        self.device = opt.device
        self.global_iter = 0

        self.dataloader_setup()
        if self.opt.dataset == 'KDD99' and self.opt.setting != 'egbad':
            self.create_D = MLP_Discriminator
            self.create_G = MLP_Generator
        elif self.opt.dataset == 'KDD99' and self.opt.setting == 'egbad':
            self.create_En = MLP_Encoder
            self.create_De = MLP_Decoder
            self.creare_D = MLP_Discriminator


        else:
            self.create_D = define_D
            self.create_G = define_G
        if self.opt.phase == 'train':
            # TODO: initialize network and optimizer
            self.generator_setup()
            self.discriminator_setup()
            if self.opt.setting != 'egbad':
                self.optims = {
                    "gen": self.optimizer_Gs,
                    "disc": self.optimizer_Ds
                }
            # TODO: define discriminator loss function
            self.l_adv = nn.BCELoss()
            self.l_con = nn.L1Loss()
            self.l_lat = l2_loss
    def dataloader_setup(self):
        raise NotImplementedError("dataloader_setup not implemented")

    def generator_setup(self):
       raise NotImplementedError("generator_setup not implemented")

    def discriminator_setup(self):
       raise NotImplementedError("discriminator_setup not implemented")

    def train_epoch(self, epoch):
        raise NotImplementedError("train_epoch not implemented")

    def test_epoch(self, epoch, plot_hist=True):
        raise NotImplementedError("test_epoch not implemented")

    def train(self):
        for net_D in self.net_Ds:
            net_D.train()
        if self.opt.setting != 'egbad':
            for net_G in self.net_Gs:
                net_G.train()
        elif self.opt.setting == 'egbad':
            for net_G in self.net_Ens:
                net_G.train()
            for net_G in self.net_Des:
                net_G.train()

        for epoch in range(self.opt.niter):
            self.train_epoch(epoch)
            self.save_weight(epoch)
            self.test_epoch(epoch)
        hist = pd.DataFrame.from_dict(self.rocs)
        hist.to_csv(
            os.path.join(self.opt.outf, self.opt.name, "{0}exp_auroc_test.csv".format(self.opt.name)))

    def save_weight(self, epoch):
        if self.opt.setting != 'egbad':
            for _idx, net_G in enumerate(self.net_Gs):
                torch.save(net_G.state_dict(), '{0}/{1}/train/weights/Net_G_{2}_epoch_{3}.pth'.format(self.opt.outf, self.opt.name,_idx, epoch))
        elif self.opt.setting == 'egbad':
            for _idx, net_G in enumerate(self.net_Ens):
                torch.save(net_G.state_dict(), '{0}/{1}/train/weights/Net_En_{2}_epoch_{3}.pth'.format(self.opt.outf, self.opt.name,_idx, epoch))
            for _idx, net_G in enumerate(self.net_Des):
                torch.save(net_G.state_dict(), '{0}/{1}/train/weights/Net_De_{2}_epoch_{3}.pth'.format(self.opt.outf, self.opt.name,_idx, epoch))


        for _idx, net_D in enumerate(self.net_Ds):
            torch.save(net_D.state_dict(), '{0}/{1}/train/weights/Net_D_{2}_epoch_{3}.pth'.format(self.opt.outf, self.opt.name,_idx, epoch))


    def load_weight(self, pathlist:dict):
        self.net_Gs = []
        self.net_Ds = []
        for weight in pathlist['net_G']:
            net_G = define_G(self.opt).to(self.device)
            net_G.load_state_dict(torch.load(weight, map_location=self.device))
            self.net_Gs.append(net_G)
        for weight in pathlist['net_D']:
            net_D = define_D(self.opt).to(self.device)
            net_D.load_state_dict(torch.load(weight, map_location=self.device))
            self.net_Ds.append(net_D)

    def get_best_result(self):
        return max(self.rocs)







