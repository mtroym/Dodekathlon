import logging
import logging.config
import ntpath
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import visdom
from torch.utils.tensorboard import SummaryWriter

from tools import html
from utils import util

logging.getLogger('PIL').setLevel(logging.INFO)


class Visualizer:
    def __init__(self, opt):
        # self.opt = opt

        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.expr_name
        self.opt = opt
        self.saved = False
        self.writer = SummaryWriter(opt.expr_dir)

        if self.display_id > 0:
            self.vis = visdom.Visdom(port=opt.display_port)
        logging.config.fileConfig(os.path.join(opt.configure_path, 'logging.conf'))
        self.logger = logging.getLogger()
        if self.use_html:
            self.web_dir = os.path.join(opt.expr_dir, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            # print('=> creating web directory:\n\t{}'.format(os.path.abspath(self.web_dir)))
            util.make_dirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.expr_dir, 'loss_log.txt')
        self.logger.addHandler(logging.FileHandler(self.log_name, mode='a', encoding=None, delay=False))
        now = time.strftime("%c")
        self.logger.info('================ Training Loss (%s) ================' % now)

    def reset(self):
        self.saved = False

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch: int, niter: int, save_result: bool = True):
        for label, image_numpy in visuals.items():
            if not isinstance(image_numpy, type(np.array([1]))):
                print(image_numpy.shape)
                visuals[label] = (image_numpy.detach().numpy().transpose([0, 2, 3, 1]) * 255).astype(np.uint8)[0]
            else:
                visuals[label] = (image_numpy.transpose([0, 2, 3, 1]) * 255).astype(np.uint8)[0]
        if self.display_id > 0:  # show images in the browser
            ncols = self.opt.display_single_pane_ncols
            if ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                image_numpy = None
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=self.display_id + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def display_vis_loss(self, res: dict, epochs: int, iters: int):
        for k, v in res["vis"].items():
            img_grid = torchvision.utils.make_grid((v + 1) / 2)
            cv2.imwrite(os.path.join(self.opt.expr_dir, "{}.png".format(iters)), img_grid.cpu().numpy().transpose([1, 2, 0])[:, :, ::-1] * 255)
            self.writer.add_image(k, img_grid, global_step=epochs)
        for k, v in res["loss"].items():
            self.writer.add_scalar(k, v, iters)

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)
        self.logger.info(message)
        # with open(self.log_name, "a") as log_file:
        #     log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.jpg' % (image_path[0], label)
            save_path = os.path.join(image_dir, image_name)
            print(save_path)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
