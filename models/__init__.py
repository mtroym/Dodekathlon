from models.base import BaseModel
from models.PATN import PATNTransferModel
from models.GMM import CTPSModel
from models.CAN import CANModel
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
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    print("=> model [{}] was created".format(model.name))
    return model
