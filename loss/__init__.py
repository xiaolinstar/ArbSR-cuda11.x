from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')


class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print("Preparing loss function:")

        self.n_GPUs = args.n_GPUs
        self.loss = list()
        self.loss_module = nn.ModuleList()

        loss_function = None
        # default loss: 1*L1
        # weight=1, loss_type=L1
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')

            if loss_type == "MSE":
                loss_function = nn.MSELoss()
            elif loss_type == "L1":
                loss_function = nn.L1Loss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
            })

            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for loss_item in self.loss:
            if loss_item['function'] is not None:
                print('{:.3f} * {}'.format(loss_item['weight'], loss_item['type']))
                self.loss_module.append(loss_item['function'])

        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)

        if args.precision == 'half':
            self.loss_module.half()

        # may lead 'ModuleList' -> 'DataParallel'
        if not args.cpu and self.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load != '.':
            self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr):
        losses = []
        for i, loss_item in enumerate(self.loss):
            if loss_item['function'] is not None:
                loss = loss_item['function'](sr, hr)
                effective_loss = loss_item['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif loss_item['type'] == 'DIS':
                self.log[-1, i] += self.loss[i-1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, 1] += loss_sum.item()

        return loss_sum

    def load(self, path, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(path, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(path, 'loss_log.pt'))

        try:
            for item in self.get_loss_module():
                if hasattr(item, 'scheduler'):
                    for _ in range(len(self.log)):
                        item.scheduler.step()
        except TypeError as e:
            print(e)

    def save(self, path):
        torch.save(self.state_dict(), os.path.join(path, 'loss.pt'))
        torch.save(self.log, os.path.join(path, 'loss_log.pt'))

    def step(self):
        try:
            for item in self.get_loss_module():
                if hasattr(item, 'scheduler'):
                    for _ in range(len(self.log)):
                        item.scheduler.step()
        except TypeError as e:
            print(e)

    def get_loss_module(self):
        # ModuleList
        if self.n_GPUs == 1 or self.cpu():
            return self.loss_module
        # DataParallel
        else:
            return self.loss_module.module

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for loss_item, log_item in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(loss_item['type'], log_item/n_samples))

        return ''.join(log)

    def plot_loss(self, path, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, loss_item in enumerate(self.loss):
            label = '{} Loss'.format(loss_item['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.pdf'.format(path, loss_item['type']))
            plt.close(fig)
