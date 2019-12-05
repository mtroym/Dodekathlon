from models.base import BaseModel
from models.PATN import PATNTransferModel


def create_model(opt):
    model = BaseModel(opt)
    if opt.model == 'Base':
        pass
    elif opt.model == 'PATN':
        model = PATNTransferModel(opt)
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    print("=> model [{}] was created".format(model.name))
    return model
