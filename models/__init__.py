from models.CAN import CANModel
from models.GANzoo.DCGAN import DCGANModel
from models.GMM import CTPSModel
from models.PATN import PATNTransferModel
from models.base import BaseModel


def create_model(opt):
    model = BaseModel(opt)
    if opt.model == 'Base':
        pass
    elif opt.model == 'PATN':
        model = PATNTransferModel(opt)
    elif opt.model == 'CTPS':
        model = CTPSModel(opt)
    elif opt.model == 'CAN':
        model = CANModel(opt)
    elif opt.model == 'DCGAN':
        model = DCGANModel(opt)
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    print("=> model [{}] was created".format(model.name))
    return model
