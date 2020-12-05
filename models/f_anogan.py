from models.basemodel import *
import torch
from models.evaluate import roc, evaluate, auprc
from models.networks import Encoder, Decoder, FDiscriminator, FMLP_Enocder, FMLP_Decoder, FMLP_Discriminator
import numpy as np
import random
from torch.autograd import Variable
import torch.autograd as autograd

class f_anogan():
    '''
    F_anogan model.
    '''
    def __init__(self, opt):
        if opt.dataset == 'KDD99':
            self.create_Ge = FMLP_Enocder
            self.create_Gd = FMLP_Decoder
            self.create_D = FMLP_Discriminator
        else:
            self.create_Ge = Encoder
            self.create_Gd = Decoder
            self.create_D = FDiscriminator

        self.opt = opt
        self.epoch = self.opt.niter
        self.visualizer = Visualizer(self.opt)
        self.device = opt.device
        self.global_iter = 0

        self.generator_setup()
        self.discriminator_setup()
        self.dataloader_setup()
        self.l_adv = nn.BCELoss()
        self.l_con = l2_loss
        self.l_lat = l2_loss

    def dataloader_setup(self):
        self.dataloader = load_data(self.opt)

    def generator_setup(self):
        self.net_Ges = []
        self.net_Gds = []
        for _ in range(0, self.opt.n_G):
            net_Ge = self.create_Ge(self.opt.isize, self.opt.nz, self.opt.nc, self.opt.ndf, self.opt.ngpu).to('cuda')
            net_Gd = self.create_Gd(self.opt.isize, self.opt.nz, self.opt.nc, self.opt.ngf, self.opt.ngpu).to('cuda')

            self.net_Ges.append(net_Ge)
            self.net_Gds.append(net_Gd)

        self.optimizer_Ge = torch.optim.Adam(sum([list(generator.parameters()) for generator in self.net_Ges],[]),
                                             lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_Gd = torch.optim.Adam(sum([list(generator.parameters()) for generator in self.net_Gds],[]),
                                             lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        alpha = torch.Tensor(np.random.random(([real_samples.size(0)]+[1 for _ in range(len(real_samples.shape)-1)] ))).to(self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates, _ = D(interpolates)
        fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(self.device)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def discriminator_setup(self):
        self.net_Ds = []
        self.optimizer_Ds = []
        for _ in range(0, self.opt.n_D):
            net_D = self.create_D(self.opt.isize, self.opt.nz, self.opt.nc, self.opt.ndf, self.opt.ngpu).to('cuda')
            # TODO: initialized weight with prior N(0, 0.02) [From bayesian GAN]
            # net_D.apply(weights_init)
            self.net_Ds.append(net_D)
        self.optimizer_D = torch.optim.Adam(sum([list(discriminator.parameters()) for discriminator in self.net_Ds],[]),
                                       lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def gan_training(self, epoch):
        for i, (imgs, _) in enumerate(self.dataloader.train):
            real_imgs = imgs.to(self.device)

            self.optimizer_D.zero_grad()

            z = torch.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.opt.nz))).to(self.device)

            i_G = random.randint(0, self.opt.n_G-1)
            i_D = random.randint(0, self.opt.n_D-1)
            fake_imgs = self.net_Gds[i_G](z)

            # Real images
            real_validity, _ = self.net_Ds[i_D](real_imgs)
            # Fake images
            fake_validity, _ = self.net_Ds[i_D](fake_imgs)
            # Gradient penalty
            gradient_penalty = self.compute_gradient_penalty( self.net_Ds[i_D], real_imgs.data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty

            d_loss.backward()
            self.optimizer_D.step()

            self.optimizer_Gd.zero_grad()

            fake_imgs =  self.net_Gds[i_G](z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images`
            fake_validity, _ = self.net_Ds[i_D](fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            self.optimizer_Gd.step()
            if i % 10 ==0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.epoch, i, len(self.dataloader.train), d_loss.item(), g_loss.item())
                )


    def enc_training(self, epoch):
        for i, (imgs, _) in enumerate(self.dataloader.train):

            i_G = random.randint(0, self.opt.n_G-1)
            i_D = random.randint(0, self.opt.n_D-1)
            imgs = imgs.to('cuda')

            z = self.net_Ges[i_G](imgs).squeeze()
            fake_imgs = self.net_Gds[i_G](z)

            _, image_feats = self.net_Ds[i_D](imgs)
            _, recon_feats = self.net_Ds[i_D](fake_imgs)

            loss_img = self.l_con(imgs, fake_imgs)
            loss_feat = self.l_lat(image_feats, recon_feats)
            loss = loss_feat+loss_img

            loss.backward()
            self.optimizer_Ge.step()
            self.optimizer_Ge.zero_grad()
            self.net_Ds[i_D].zero_grad()
            if i % 10 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [Ge %d loss: %f]"#, [D %d loss: %f]"
                    % (epoch, self.epoch, i, len(self.dataloader.train), i_G,  loss.item())#, i_D, loss_feat.item())
                )

    def train(self):
        for epoch in range(self.opt.niter):
            self.gan_training(epoch)

        for i_G in range(0, self.opt.n_G):
            self.net_Gds[i_G].eval()

        for epoch in range(self.opt.niter //5):
            self.enc_training(epoch)
            self.test_epoch(epoch)


    def test_epoch(self, epoch, plot_hist=True):
        with torch.no_grad():
            scores = torch.empty(
                size=(len(self.dataloader.valid.dataset), self.opt.n_G, self.opt.n_D),
                dtype=torch.float32,
                device='cuda')
            labels = torch.zeros(size=(len(self.dataloader.valid.dataset),),
                                        dtype=torch.long, device='cuda')
            i_score = 0.
            f_score = 0.
            for i, (imgs, lbls) in enumerate(self.dataloader.valid):
                imgs = imgs.to('cuda')
                lbls = lbls.to('cuda')

                labels[i*self.opt.batchsize:(i+1)*self.opt.batchsize].copy_(lbls)
                recon_dims = [1,2,3]
                feat_dims = [1]
                for i_G in range(self.opt.n_G):
                    for i_D in range(self.opt.n_D):
                        # the commented code is for oct only
                        if self.opt.dataset == 'OCT':
                            recon_dims = [1,2,3,4]
                            feat_dims = [1,2]
                            imgs = imgs.view(-1, self.opt.nc, self.opt.isize, self.opt.isize) # view (batch, patch, nc, w, h) -> (batch*patch, nc, w, h)
                        emb_query = self.net_Ges[i_G](imgs)
                        fake_imgs = self.net_Gds[i_G](emb_query)
                        # emb_query = emb_query.view(batch, -1, nz)

                        _, image_feats  = self.net_Ds[i_D](imgs)
                        _, recon_feats = self.net_Ds[i_D](fake_imgs)

                        if self.opt.dataset == 'OCT':
                            emb_query = emb_query.view(self.opt.batchsize, -1, self.opt.nz)
                            imgs = imgs.view(self.opt.batchsize, -1, self.opt.nc, self.opt.isize, self.opt.isize)
                            fake_imgs = fake_imgs.view(self.opt.batchsize, -1, self.opt.nc, self.opt.isize, self.opt.isize)

                        recon_distance = torch.mean(torch.pow(imgs-fake_imgs, 2), dim=recon_dims)
                        feat_distance = torch.mean(torch.pow(image_feats-recon_feats, 2), dim=feat_dims)
                        score = 0.9 * recon_distance+0.1*feat_distance
                        scores[i*self.opt.batchsize:(i+1)*self.opt.batchsize, i_G, i_D].copy_(score)

            labels = labels.cpu()
            scores = torch.mean(scores, dim=[1,2])
            scores = scores.cpu().squeeze()

            hist = {}
            hist['scores'] = scores
            hist['labels'] = labels
            import pandas as pd
            hist = pd.DataFrame.from_dict(hist)
            auroc = roc(labels, scores, epoch)
            print('roc:', auroc)
            hist.to_csv(os.path.join(self.opt.outf, self.opt.name,
                                     "{0}exp_{1}epoch_score_train.csv".format(self.opt.name, epoch)))
