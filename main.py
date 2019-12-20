import importlib
import time

import cv2

from tools.Saver import Saver
from tools.Visualizer import Visualizer


def init_all():
    print("=> initializing. parsing arguments.")
    opts = importlib.import_module('options.custom-options')
    opt_init = opts.TrainOptions().opt

    data_lib = importlib.import_module('datasets')
    models_lib = importlib.import_module('models')
    criterions_lib = importlib.import_module('criterions')

    dataloader_init = data_lib.create_dataloader(opt_init)
    model_init = models_lib.create_model(opt_init)
    loss_init, metrics_init = criterions_lib.create_criterion(opt_init)
    return opt_init, dataloader_init, model_init, loss_init, metrics_init


if __name__ == '__main__':
    opt, dataloader, model, loss, metrics = init_all()
    visualizer = Visualizer(opt)
    saver = Saver(opt)
    saver.latest()
    for epoch in range(0, 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataloader):
            iter_start_time = time.time()
            vis = model.train_batch(inputs=data, loss=loss, metrics=metrics)
            # visualizer.display_current_results(vis, epoch, save_result=True)
        #
        # for i, data in enumerate(dataloader_val):
        #     iter_start_time

            #         visualizer.reset()
            # pass
            # cv2.imwrite("test.png", (data["Source"][0].data.numpy().transpose([1, 2, 0]) + 0.5) * 255)
            # cv2.imwrite("testkp.png", data["Source_KP"][0].data.numpy().sum(axis=-1) * 255)
    #
    #         # todo: add this.
    #         model.set_input(data)
    #         model.optimize_parameters()
    #
    #         if epoch % opt.display_freq == 0:
    #             save_result = epoch % opt.update_html_freq == 0
    #             visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
    #
    #         if epoch % opt.print_freq == 0:
    #             errors = model.get_current_errors()
    #             t = (time.time() - iter_start_time) / opt.batchSize
    #             visualizer.print_current_errors(epoch, epoch_iter, errors, t)
    #             if opt.display_id > 0:
    #                 visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)
    #
    #         if epoch % opt.save_latest_freq == 0:
    #             print('saving the latest model (epoch %d, total_steps %d)' %
    #                   (epoch, epoch))
    #             model.save('latest')
    #
    #     if epoch % opt.save_epoch_freq == 0:
    #         print('saving the model at the end of epoch %d, iters %d' %
    #               (epoch, epoch))
    #         model.save('latest')
    #         model.save(epoch)
    #
    #     print('End of epoch %d / %d \t Time Taken: %d sec' %
    #           (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    #     model.update_learning_rate()
