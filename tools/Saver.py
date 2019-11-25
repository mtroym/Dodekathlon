import logging
import os

import torch

logging = print


class Saver:
    def __init__(self, opt):
        self.opt = opt
        self.loaded = {'epoch'     : -1, 'model_file': None, 'criterion_file': None,
                       'optim_file': None}

    def latest(self):
        latest_path = os.path.join(self.opt.resume, 'latest.pth.tar')
        if not os.path.exists(latest_path):
            return None
        logging.info('=> Loading the latest checkpoint ' + latest_path)
        return torch.load(latest_path)

    def best(self):
        best_path = os.path.join(self.opt.resume, 'best.pth.tar')
        if not os.path.exists(best_path):
            return None
        logging.info('=> Loading the best checkpoint ' + best_path)
        return torch.load(best_path)

    def load(self):
        logging.info(self.opt.resume, self.opt.epoch_num)
        epoch = self.opt.epoch_num
        if epoch == 0:
            return None
        elif epoch == -1:
            return self.latest()
        elif epoch == -2:
            return self.best()
        else:
            model_file = 'model_' + str(epoch) + '.pth.tar'
            criterion_file = 'criterion_' + str(epoch) + '.pth.tar'
            optim_file = 'optimState_' + str(epoch) + '.pth.tar'
            loaded = {'epoch'     : epoch, 'model_file': model_file, 'criterion_file': criterion_file,
                      'optim_file': optim_file}
            return loaded

    def save(self, epoch, model, criterion, optimizer, best_model, loss, opt):
        logging.info('=> Saving checkpoints...')
        # if isinstance(model, nn.DataParallel):
        #     model = model.get(0)
        # TODO

        model_file = 'model_' + str(epoch) + '.pth.tar'
        criterion_file = 'criterion_' + str(epoch) + '.pth.tar'
        optim_file = 'optimState_' + str(epoch) + '.pth.tar'

        if best_model or (epoch % self.opt.save_epoch == 0):
            torch.save(model.state_dict(), os.path.join(opt.resume, model_file))
            torch.save(criterion, os.path.join(opt.resume, criterion_file))
            torch.save(optimizer.state_dict(), os.path.join(opt.resume, optim_file))
            info = {'epoch'     : epoch, 'model_file': model_file, 'criterion_file': criterion_file,
                    'optim_file': optim_file, 'loss': loss}
            torch.save(info, os.path.join(opt.resume, 'latest.pth.tar'))

        if best_model:
            info = {'epoch'     : epoch, 'model_file': model_file, 'criterion_file': criterion_file,
                    'optim_file': optim_file, 'loss': loss}
            torch.save(info, os.path.join(self.opt.resume, 'best.pth.tar'))
            torch.save(model.state_dict(), os.path.join(self.opt.resume, 'model_best.pth.tar'))
