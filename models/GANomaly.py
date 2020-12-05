from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from models.networks import NetG, NetD, weights_init
from models.evaluate import evaluate
from utils.visualizer import Visualizer
from utils.loss import l2_loss

import random






class BaseModel():
    """ Base Model for ganomaly
    """

    def __init__(self, opt, dataloader):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")

    ##
    def set_input(self, input: torch.Tensor):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())

            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])

    ##
    def seed(self, seed_value):
        """ Seed

        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([
            ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_enc', self.err_g_enc.item())])

        return errors

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netgs[0](self.fixed_input)[0].data

        return reals, fakes, fixed

    ##
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)
        for idx_g in range(self.opt.NG):
            torch.save({'epoch': epoch + 1, 'state_dict': self.netgs[idx_g].state_dict()},
                       '%s/netG_%d.pth' % (weight_dir, idx_g))
        for idx_d in range(self.opt.ND):
            torch.save({'epoch': epoch + 1, 'state_dict': self.netds[idx_d].state_dict()},
                       '%s/netD_%d.pth' % (weight_dir, idx_d))

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """

        # self.netg.train()

        epoch_iter = 0
        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            idx_d = random.randint(0, self.opt.ND - 1)
            idx_g = random.randint(0, self.opt.NG - 1)

            self.netgs[idx_g].train()
            self.netds[idx_d].train()
            self.set_input(data)
            # self.optimize()
            self.optimize_params(idx_d, idx_g)

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)


        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch + 1, self.opt.niter))

    ##
    def train(self):
        """ Train the model
        """

        self.total_steps = 0
        best_auc = 0

        # Train for niter epochs.
        print(">> Training model %s." % self.name)
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_one_epoch()
            res = self.test()
            if res['AUC'] > best_auc:
                best_auc = res['AUC']
                self.save_weights(self.epoch)
            self.visualizer.print_current_performance(res, best_auc)
        print(">> Training model %s.[Done]" % self.name)

    ##
    def test(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32,
                                         device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,
                                         device=self.device)
            self.latent_i = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32,
                                        device=self.device)
            self.latent_o = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32,
                                        device=self.device)


            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.dataloader['test'], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                latent_is = []
                latent_os = []
                for idx_g in range(self.opt.NG):
                    # self.fake, latent_i, latent_o = self.netgs[idx_g](self.input)
                    _, latent_i, latent_o = self.netgs[idx_g](self.input)
                    latent_is.append(latent_i)
                    latent_os.append(latent_o)

                latent_i = torch.mean(torch.stack(latent_is), dim=0)
                latent_o = torch.mean(torch.stack(latent_os), dim=0)

                error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)
                time_o = time.time()

                self.an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error.reshape(
                    error.size(0))
                self.gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = self.gt.reshape(
                    error.size(0))
                self.latent_i[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_i.reshape(
                    error.size(0), self.opt.nz)
                self.latent_o[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_o.reshape(
                    error.size(0), self.opt.nz)

                self.times.append(time_o - time_i)


            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (
                        torch.max(self.an_scores) - torch.min(self.an_scores))
            # auc, eer = roc(self.gt_labels, self.an_scores)
            auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance


##
class Ganomaly(BaseModel):
    """GANomaly Class
    """

    @property
    def name(self):
        return 'Ganomaly'

    def __init__(self, opt, dataloader):
        super(Ganomaly, self).__init__(opt, dataloader)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netgs = [NetG(self.opt).to(self.device) for _ in range(self.opt.n_G)]
        self.netds = [NetD(self.opt).to(self.device) for _ in range(self.opt.n_D)]
        for i in range(self.opt.n_G):
            self.netgs[i].apply(weights_init)
        for i in range(self.opt.n_D):
            self.netds[i].apply(weights_init)


        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                 device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize),
                                       dtype=torch.float32, device=self.device)
        self.real_label = torch.ones(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        ##
        # Setup optimizer
        if self.opt.isTrain:
            for i in range(self.opt.n_G):
                self.netgs[i].train()
            for i in range(self.opt.n_D):
                self.netds[i].train()
            self.optimizer_d = optim.Adam(sum([list(self.netds[i].parameters()) for i in range(self.opt.ND)], []),
                                          lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(sum([list(self.netgs[i].parameters()) for i in range(self.opt.NG)], []),
                                          lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    ##
    def forward_g(self, idx_g):
        """ Forward propagate through netG
        """
        self.fake, self.latent_i, self.latent_o = self.netgs[idx_g](self.input)

    ##
    def forward_d(self, idx_d):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netds[idx_d](self.input)
        self.pred_fake, self.feat_fake = self.netds[idx_d](self.fake.detach())

    ##
    def backward_g(self, idx_d):
        """ Backpropagate through netG
        """
        self.err_g_adv = self.l_adv(self.netds[idx_d](self.input)[1], self.netds[idx_d](self.fake)[1])
        self.err_g_con = self.l_con(self.fake, self.input)
        self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)
        self.err_g = self.err_g_adv * self.opt.w_adv + \
                     self.err_g_con * self.opt.w_con + \
                     self.err_g_enc * self.opt.w_enc
        self.err_g.backward(retain_graph=True)

    ##
    def backward_d(self):
        """ Backpropagate through netD
        """
        # Real - Fake Loss
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)

        # NetD Loss & Backward-Pass
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()

    ##
    def reinit_d(self, idx_d):
        """ Re-initialize the weights of netD
        """
        self.netds[idx_d].apply(weights_init)
        print('   Reloading net d')

    def optimize_params(self, idx_d, idx_g):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g(idx_g)
        self.forward_d(idx_d)

        # Backward-pass
        # netg
        self.optimizer_g.zero_grad()
        self.backward_g(idx_d)
        self.optimizer_g.step()

        # netd
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
        # if self.err_d.item() < 1e-5: self.reinit_d(idx_d)
