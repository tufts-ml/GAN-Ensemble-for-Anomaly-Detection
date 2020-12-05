from models.basemodel import *
import torch
from models.evaluate import roc, evaluate, auprc
from models.networks import Encoder, Decoder, EGBADDiscriminator, MLP_Encoder, MLP_Decoder, MLP_Discriminator


class egbad_kdd(ANBase):
    '''
    EGBAD model for KDD dataset.
    '''
    def __init__(self, opt):
        super(egbad_kdd, self).__init__(opt)
        self.rocs = {'auroc': []}

    def dataloader_setup(self):
        self.dataloader = {
            'gen': [load_data(self.opt) for _ in range(self.opt.n_G)],
            'disc': [load_data(self.opt) for _ in range(self.opt.n_D)]
        }

    def generator_setup(self):
        self.net_Ens = []
        self.net_Des = []
        self.optimizer_Ens = []
        self.optimizer_Des = []
        for _idxmc in range(0, self.opt.n_G):
            net_En = MLP_Encoder(self.opt).to('cuda')
            net_De = MLP_Decoder(self.opt).to('cuda')
            # TODO: initialized weight with prior N(0, 0.02) [From bayesian GAN]
            net_En.apply(weights_init)
            net_De.apply(weights_init)
            optimizer_En = torch.optim.Adam(net_En.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            optimizer_De = torch.optim.Adam(net_De.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.net_Ens.append(net_En)
            self.net_Des.append(net_De)
            self.optimizer_Ens.append(optimizer_En)
            self.optimizer_Des.append(optimizer_De)

    def discriminator_setup(self):
        self.net_Ds = []
        self.optimizer_Ds = []
        for _idxmc in range(0, self.opt.n_D):
            net_D = MLP_Discriminator(self.opt).to('cuda')
            # TODO: initialized weight with prior N(0, 0.02) [From bayesian GAN]
            net_D.apply(weights_init)
            optimizer_D = torch.optim.Adam(net_D.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.net_Ds.append(net_D)
            self.optimizer_Ds.append(optimizer_D)

    def train_epoch(self, epoch):

        for _ in tqdm(range(len(self.dataloader["gen"][0].train)), leave=False,
                      total=len(self.dataloader["gen"][0].train)):
            self.global_iter += 1

            # TODO update each disc with all gens
            for _idxD in range(self.opt.n_D):
                x_real, _ = next(iter(self.dataloader["disc"][_idxD].train))  # get data batch
                x_real = x_real.to(self.device)
                self.net_Ds[_idxD].zero_grad()
                label_real = torch.ones(x_real.shape[0]).to(self.device)  # create real label

                err_d_fakes = 0.0
                err_d_reals = 0.0

                for _idxG in range(self.opt.n_G):
                    z_gen = self.net_Ens[_idxG](x_real)

                    z = torch.randn(x_real.shape[0], self.opt.nz).to(self.device)


                    x_fake = self.net_Des[_idxG](z)  # get fake image from network G


                    pred_encoder, feat_encoder = self.net_Ds[_idxD](z_gen.detach(), x_real)
                    pred_decoder, feat_decoder = self.net_Ds[_idxD](z, x_fake.detach())


                    label_fake = torch.zeros(x_real.shape[0]).to(self.device)

                    err_d_fake = self.l_adv(pred_decoder, label_fake)
                    err_d_real = self.l_adv(pred_encoder, label_real)

                    err_d_fakes += err_d_fake
                    err_d_reals += err_d_real

                err_d_total_loss = err_d_fakes + err_d_reals

                err_d_total_loss /= self.opt.n_G

                err_d_total_loss.backward()
                self.optimizer_Ds[_idxD].step()

            # TODO update each gen with all discs
            for _idxG in range(self.opt.n_G):
                x_real, _ = next(iter(self.dataloader["gen"][_idxG].train))  # get data batch
                x_real = x_real.to(self.device)
                self.net_Ens[_idxG].zero_grad()
                self.net_Des[_idxG].zero_grad()

                z = torch.randn(x_real.shape[0], self.opt.nz).to(self.device)

                x_fake = self.net_Des[_idxG](z)
                z_gen = self.net_Ens[_idxG](x_real)


                err_g_fakes = 0.0
                err_g_reals = 0.0

                for _idxD in range(self.opt.n_D):
                    pred_encoder, feat_encoder = self.net_Ds[_idxD](z_gen, x_real)
                    pred_decoder, feat_decoder = self.net_Ds[_idxD](z, x_fake)

                    label_real = torch.ones(x_real.shape[0]).to(self.device)  # create inversed label

                    label_fake = torch.zeros(x_real.shape[0]).to(self.device)

                    err_g_fake = self.l_adv(pred_decoder, label_real)
                    err_g_real = self.l_adv(pred_encoder, label_fake)  # strange
                    err_g_fakes += err_g_fake
                    err_g_reals += err_g_real

                err_en_total_loss = err_g_reals / self.opt.n_D
                err_de_total_loss = err_g_fakes / self.opt.n_D

                err_en_total_loss.backward()
                err_de_total_loss.backward()

                self.optimizer_Ens[_idxG].step()
                self.optimizer_Des[_idxG].step()


    def test_epoch(self, epoch, plot_hist=True):
        """ Test eEGBAD model."""

        with torch.no_grad():


            self.opt.phase = 'test'
            self.l_adv_t = torch.nn.BCELoss(reduction = 'none')



            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(
                size=(self.opt.n_G * self.opt.n_D, len(self.dataloader["gen"][0].valid.dataset)),
                dtype=torch.float32,
                device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader["gen"][0].valid.dataset),), dtype=torch.long,
                                         device=self.device)
            self.features = torch.zeros(size=(len(self.dataloader["gen"][0].valid.dataset), self.opt.nz),
                                        dtype=torch.float32,
                                        device=self.device)

            ensemble_iter = 0
            for g in range(self.opt.n_G):
                for d in range(self.opt.n_D):

                    self.total_steps = 0
                    epoch_iter = 0
                    for i, (x_real, label) in enumerate(self.dataloader["gen"][0].valid, 0):
                        self.total_steps += self.opt.batchsize
                        epoch_iter += self.opt.batchsize

                        # Forward - Pass
                        self.input = x_real.to(self.device)

                        z_gen = self.net_Ens[g](self.input)
                        x_fake = self.net_Des[g](z_gen)

                        pred_encoder, feat_encoder = self.net_Ds[d](z_gen, self.input)
                        pred_decoder, feat_decoder = self.net_Ds[d](z_gen, x_fake)


                        label_real = torch.ones(self.input.shape[0]).to(self.device)




                        fm = self.l_adv_t(pred_encoder, label_real)

                        error = fm

                        self.an_scores[ensemble_iter,
                        i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error
                        self.gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = label.reshape(
                            error.size(0))

                    ensemble_iter = ensemble_iter + 1

            self.an_scores = torch.mean(self.an_scores, dim=0)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / \
                             (torch.max(self.an_scores) - torch.min(self.an_scores))

            per_scores = self.an_scores.cpu()

        if plot_hist:
            self.gt_labels = self.gt_labels.cpu()
            auroc = roc(self.gt_labels, per_scores, epoch)
            precision = auprc(self.gt_labels, per_scores)

            recall = evaluate(self.gt_labels, per_scores, metric='auprc')

            print('auroc is {}'.format(auroc))

            print('recall is {}'.format(recall))
            print('precision is {}'.format(precision))

            self.rocs['auroc'].append(auroc)

            plt.ion()
            # Create data frame for scores and labels.
            scores = {}
            scores['scores'] = per_scores
            scores['labels'] = self.gt_labels

            hist = pd.DataFrame.from_dict(scores)
            hist.to_csv(os.path.join(self.opt.outf, self.opt.name,
                                     "{0}exp_{1}epoch_score_train.csv".format(self.opt.name, epoch)))