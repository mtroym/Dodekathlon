from models.base import BaseModel


def create_model(opt):
    model = BaseModel(opt)
    if opt.model == 'Base':
        pass
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    print("=> model [{}] was created".format(model.name))
    return model