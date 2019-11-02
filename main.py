import importlib

opts = importlib.import_module('options.custom-options')

if __name__ == '__main__':
    opt = opts.TrainOptions()
    print("=> initializing. parsing arguments.")
