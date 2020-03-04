import yaml

from options.base import BaseOptions


class CustomOptions(BaseOptions):
    def __init__(self):
        super(CustomOptions, self).__init__()
        BaseOptions.initialize(self)
        configure_stream = open(self.opt.configure_file, 'r')
        experiment_config = yaml.load(configure_stream)
        if "mode" not in experiment_config:
            raise RuntimeError("invalid mode in setting.")
        mode = experiment_config["mode"]

        experiment_config = experiment_config['{}_settings'.format(mode)]
        self.opt.__setattr__("mode", mode)
        self.opt = self.parse(experiment_config)
