from models.basemodel import *
import torch
from models.evaluate import roc

class model_skipgan(ANBase):
    def __init__(self, opt):
        super(model_skipgan, self).__init__(opt)
        self.rocs = {'auroc':[]}

    def dataloader_setup(self):
        self.dataloader = {
            'gen': [load_data(self.opt) for _ in range(self.opt.n_G)],
            'disc': [load_data(self.opt) for _ in range(self.opt.n_D)]
        }

    def generator_setup(self):
        self.net_Gs = []
        self.optimizer_Gs = []
        for _idxmc in range(0, self.opt.n_G):
            net_G = self.create_G(self.opt).to('cuda')
            # TODO: initialized weight with prior N(0, 0.02) [From bayesian GAN]
            # net_G.apply(weights_init)
            optimizer_G = torch.optim.Adam(net_G.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.net_Gs.append(net_G)
            self.optimizer_Gs.append(optimizer_G)

    def discriminator_setup(self):
        self.net_Ds = []
        self.optimizer_Ds = []
        for _idxmc in range(0, self.opt.n_D):
            net_D = self.create_D(self.opt).to('cuda')
            # TODO: initialized weight with prior N(0, 0.02) [From bayesian GAN]
            # net_D.apply(weights_init)
            optimizer_D = torch.optim.Adam(net_D.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.net_Ds.append(net_D)
            self.optimizer_Ds.append(optimizer_D)

    def train_epoch(self, epoch):

        for _ in tqdm(range(len(self.dataloader["gen"][0].train)), leave=False,
                          total=len(self.dataloader["gen"][0].train)):
            self.global_iter += 1

            # TODO update each disc with all gens
            for _idxD in range(self.opt.n_D):
                x_real, _ = next(iter(self.dataloader["disc"][_idxD].train))    # get data batch
                x_real = x_real.to(self.device)
                self.net_Ds[_idxD].zero_grad()
                label_real = torch.ones(x_real.shape[0]).to(self.device)        # create real label
                pred_real, feat_real = self.net_Ds[_idxD](x_real)               # get real prediction from network D
                err_d_real = self.l_adv(pred_real, label_real)

                err_d_fakes = 0.0
                err_d_lats = 0.0

                for _idxG in range(self.opt.n_G):
                    x_fake = self.net_Gs[_idxG](x_real)                         # get fake image from network G
                    pred_fake, feat_fake = self.net_Ds[_idxD](x_fake.detach())  # get fake prediction from network D
                    label_fake = torch.zeros(x_real.shape[0]).to(self.device)

                    err_d_fake = self.l_adv(pred_fake, label_fake)
                    err_d_lat = self.l_lat(feat_fake,feat_real)

                    err_d_fakes += err_d_fake
                    err_d_lats += err_d_lat

                err_d_total_loss = err_d_fakes + err_d_lats + err_d_real * self.opt.n_G

                err_d_total_loss /= self.opt.n_G

                err_d_total_loss.backward()
                self.optims['disc'][_idxD].step()

            # TODO update each gen with all discs
            for _idxG in range(self.opt.n_G):
                x_real, _ = next(iter(self.dataloader["gen"][_idxG].train))     # get data batch
                x_real = x_real.to(self.device)
                self.net_Gs[_idxG].zero_grad()
                x_fake = self.net_Gs[_idxG](x_real)                             # get fake image from network G
                err_g_con = self.l_con(x_fake, x_real)                          # get reconstruction loss



                err_g_fakes = 0.0
                err_g_lats = 0.0

                for _idxD in range(self.opt.n_D):
                    pred_real, feat_real = self.net_Ds[_idxD](x_real)
                    pred_fake, feat_fake = self.net_Ds[_idxD](x_fake)           # get fake prediction from network D
                    label_real = torch.ones(x_real.shape[0]).to(self.device)    # create inversed label

                    err_g_fake = self.l_adv(pred_fake, label_real)
                    err_g_lat = self.l_lat(feat_fake, feat_real)
                    err_g_fakes += err_g_fake
                    err_g_lats += err_g_lat

                err_g_total_loss = err_g_fakes+ err_g_lats + self.opt.w_con*err_g_con* self.opt.n_D
                err_g_total_loss /= self.opt.n_D

                err_g_total_loss.backward()
                self.optims['gen'][_idxG].step()


    def test_epoch(self,epoch, plot_hist=True):

            with torch.no_grad():
                # Load the weights of netg and netd.
                if self.opt.load_weights:
                    self.load_weights(is_best=True)

                self.opt.phase = 'test'

                scores = {}

                # Create big error tensor for the test set.
                self.an_scores = torch.zeros(size=(9,len(self.dataloader["gen"][0].valid.dataset)), dtype=torch.float32,
                                             device=self.device)
                self.gt_labels = torch.zeros(size=(len(self.dataloader["gen"][0].valid.dataset),), dtype=torch.long, device=self.device)
                self.features = torch.zeros(size=(len(self.dataloader["gen"][0].valid.dataset), self.opt.nz), dtype=torch.float32,
                                            device=self.device)


                ensemble_iter = 0
                for g in range(3):
                    for d in range(3):

                      self.total_steps = 0
                      epoch_iter = 0
                      for i, (x_real, label) in enumerate(self.dataloader["gen"][0].valid, 0):
                          self.total_steps += self.opt.batchsize
                          epoch_iter += self.opt.batchsize


                          # Forward - Pass
                          self.input = x_real.to(self.device)
                          self.fake = self.net_Gs[g](self.input)

                          _, self.feat_real = self.net_Ds[d](self.input)
                          _, self.feat_fake = self.net_Ds[d](self.fake)

                          # Calculate the anomaly score.
                          si = self.input.size()
                          sz = self.feat_real.size()
                          rec = (self.input - self.fake).view(si[0], si[1] * si[2] * si[3])
                          lat = (self.feat_real - self.feat_fake).view(sz[0], sz[1] * sz[2] * sz[3])
                          rec = torch.mean(torch.pow(rec, 2), dim=1)
                          lat = torch.mean(torch.pow(lat, 2), dim=1)
                          error = self.opt.alpha * rec + (1-self.opt.alpha) * lat



                          self.an_scores[ensemble_iter , i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error.reshape(
                              error.size(0))
                          self.gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = label.reshape(
                              error.size(0))

                      ensemble_iter =  ensemble_iter + 1



                self.an_scores = torch.mean(self.an_scores, dim = 0)

                # Scale error vector between [0, 1]
                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / \
                                 (torch.max(self.an_scores) - torch.min(self.an_scores))

                per_scores = self.an_scores.cpu().squeeze()

            if plot_hist:

                self.gt_labels = self.gt_labels.cpu()
                auroc = roc(self.gt_labels, per_scores, epoch)

                print('auroc is {}'.format(auroc))
                self.rocs['auroc'].append(auroc)
               
                plt.ion()
                # Create data frame for scores and labels.
                scores = {}
                scores['scores'] = per_scores
                scores['labels'] = self.gt_labels

                hist = pd.DataFrame.from_dict(scores)
                hist.to_csv(os.path.join(self.opt.outf, self.opt.name,
                                         "{0}exp_{1}epoch_score_train.csv".format(self.opt.name, epoch)))