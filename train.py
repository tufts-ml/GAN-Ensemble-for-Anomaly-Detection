from options import Options, setup_dir
from models.basemodel import ANBase
from models.skipgan_mxn import model_skipgan
from models.egbad_mxn import egbad_mxn
from models.egbad_KDD99 import egbad_kdd
from models.f_anogan import f_anogan
from models.GANomaly import Ganomaly


def main():
    opt = Options().parse()
    setup_dir(opt)
    if opt.setting == "skipgan":
        model = model_skipgan(opt)
    elif opt.setting == 'egbad':
        if opt.dataset != 'KDD99':
            model = egbad_mxn(opt)
        else:
            model = egbad_kdd(opt)
    elif opt.setting == 'f-anogan':
        model = f_anogan(opt)
    elif opt.setting == 'ganomaly':
        model = Ganomaly(opt)
    model.train()




if __name__ == '__main__':
    main()
