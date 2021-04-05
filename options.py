import argparse
import os
import torch


class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """ 

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--dataset', default='cifar10', help='folder | cifar10 | mnist | OCT | KDD99')
        self.parser.add_argument('--dataroot', default='', help='path to dataset')
        self.parser.add_argument('--path', default='', help='path to the folder or image to be predicted.')
        self.parser.add_argument('--setting', default='egbad',
                                 help='skipgan | egbad| f-anogan|ganomaly, type of base models to ensemble')
        self.parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
        self.parser.add_argument('--split', type=int, default=1, help='number of forward pass before backprop')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
        self.parser.add_argument('--droplast', action='store_true', default=True, help='Drop last batch size.')
        self.parser.add_argument('--isize', type=int, default=32, help='input image size.')
        self.parser.add_argument('--nc', type=int, default=1, help='input image channels')
        self.parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--ndf', type=int, default=64)
        self.parser.add_argument('--extralayers', type=int, default=0, help='Number of extra layers on gen and disc')
        self.parser.add_argument('--device', type=str, default='cuda', help='Device: cuda | cpu')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--name', type=str, default='speed', help='name of the experiment')
        self.parser.add_argument('--model', type=str, default='f-anogan', help='chooses which model to use. ganomaly. egbad')
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display', default=False, action='store_true', help='Use visdom.')
        self.parser.add_argument('--verbose', action='store_true', help='Print the training and model details.')
        self.parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
        self.parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')
        self.parser.add_argument('--abnormal_class', default='0', help='Anomaly class idx for mnist and cifar datasets')
        self.parser.add_argument('--metric', type=str, default='roc', help='Evaluation metric: roc | auprc')
        self.parser.add_argument('--bayes', action='store_true', default=False, help='Drop last batch size.')
        self.parser.add_argument('--n_G', type=int, default=3, help='number of Generator parameters')
        self.parser.add_argument('--n_D', type=int, default=3, help='number of Discriminator parameters')
        self.parser.add_argument('--save_weight', action='store_true', default=True, help='Save weight in each iteration')
        self.parser.add_argument('--arch', default='UNET', help='DCGAN | UNET')

        self.parser.add_argument('--use_2disc', action='store_true', help='Use two discriminator')
        self.parser.add_argument('--alpha', default=0.9, help='reconstruction rate for anomaly score')
        self.parser.add_argument('--n_cpu', default=5, help='cores for dataloader')
        ##
        # Train
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_image_freq', type=int, default=100, help='frequency of saving real and fake images')
        self.parser.add_argument('--save_test_images', action='store_true', help='Save test images for demo.')
        self.parser.add_argument('--load_weights', action='store_true', help='Load the pretrained weights')
        self.parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--iter', type=int, default=0, help='Start from iteration i')
        self.parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam--0.0002')

        self.parser.add_argument('--sigma_lat', type=float, default=1, help='Weight for latent space loss. default=1')
        self.parser.add_argument('--scale_con', type=float, default=0.02, help='Weight for reconstruction loss. default=0.02')
        self.parser.add_argument('--w_adv', type=float, default=1, help='Weight for adversarial loss. default=1')
        self.parser.add_argument('--w_con', type=float, default=50, help='Reconstruction loss weight')
        self.parser.add_argument('--w_enc', type=float, default=1, help='Encoder loss weight.')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        self.isTrain = True
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """
        self.opt, _ = self.parser.parse_known_args()
        self.opt.isTrain = self.isTrain   # train or test
        args = vars(self.opt)

        if self.opt.verbose:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

        return self.opt

    def parse_from_file(self, file):
        opts = open(file).readlines()[1:-1]
        opts = [option.strip().split(': ') for option in opts if len(option.strip().split(': '))==2]
        opts = [item for a in opts for item in a]
        for i, opt in enumerate(opts):
            if i % 2 == 0:
                opts[i] = '--'+opts[i]
        self.opt, _ = self.parser.parse_known_args(opts)
        self.opt.device = 'cuda' if self.opt.device == 'gpu' else 'cpu'
        self.opt.isTrain = self.isTrain  # train or test
        args = vars(self.opt)
        if self.opt.verbose:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

        return self.opt
def setup_dir(opt):
    if opt.name == 'experiment_name':
        opt.name = "%s/%s" % (opt.model, opt.dataset)
    expr_dir = os.path.join(opt.outf, opt.name, 'train')
    test_dir = os.path.join(opt.outf, opt.name, 'test')

    if not os.path.isdir(expr_dir):
        os.makedirs(expr_dir)
        os.makedirs("{0}/weights".format(expr_dir))
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)
        os.makedirs("{0}/plots".format(test_dir))

    file_name = os.path.join(expr_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(vars(opt).items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
    return opt

