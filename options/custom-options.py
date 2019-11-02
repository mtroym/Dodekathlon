import yaml

from options.base import BaseOptions


class CustomOptions(BaseOptions):
    def __init__(self, name="train"):
        super(CustomOptions, self).__init__()
        BaseOptions.initialize(self)
        configure_stream = open(self.opt.configure_file, 'r')
        experiment_config = yaml.load(configure_stream)['{}_settings'.format(name)]
        self.opt = self.parse(experiment_config)


class TrainOptions(CustomOptions):
    def __init__(self):
        CustomOptions.__init__(self, name="train")


class TestOptions(CustomOptions):
    def __init__(self):
        CustomOptions.__init__(self, name="test")

