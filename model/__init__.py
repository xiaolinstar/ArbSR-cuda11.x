import os
from importlib import import_module

import torch
from torch import nn


# 设置模块的一系列参数
# 导入模块为arbrcan
class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print("Making model...")
        try:
            self.scale_1 = args.scale_1
            self.scale_2 = args.scale_2
            self.idx_scale = 0
            self.self_ensemble = args.self_ensemble
            self.chop = args.chop
            self.precision = args.precision
            self.cpu = args.cpu
            self.device = torch.device('cpu' if args.cpu else 'cuda')
            self.n_GPUs = args.n_GPUs
            self.save_models = args.save_models
        except AttributeError as e:
            print(e)

        # 导入模块 arbrcan
        module = import_module('model.' + args.model.lower())
        # 可以直接相对路径导入

        self.model = module.make_model(args).to(self.device)

        # 半精度可以提升计算效率
        if self.precision == 'half':
            self.model.half()

        # 如果有多个GPU，则可以使用数据并行
        if not self.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(self.n_GPUs))

        # （从本地文件）记载模型参数
        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )

    def forward(self, x):
        if self.self_ensemble and not self.training:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            return self.forward_x8(x, forward_function)
        elif self.chop and not self.training:
            return self.forward_chop(x)
        else:
            return self.model(x)

    def get_model(self):
        # 单个GPU或cpu
        if self.n_GPUs <= 1 or self.cpu:
            return self.model
        # 多个GPU
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, path, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(),
            os.path.join(path, 'model', 'model_latest.pt')
        )

        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(path, 'model', 'model_best.pt')
            )

        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(path, 'model', 'model_{}.pt'.format(epoch))
            )

    def load(self, path, pre_train='.', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        # 加载预训练的参数
        # default
        if resume == 0:
            pretrained_dict = torch.load(pre_train)
            model_dict = self.get_model().state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.get_model().load_state_dict(model_dict)
            print('load from RCAN_BIX4 pre-trained model')

        elif resume > 0:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(path, 'model', 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=False
            )
            print('load from model_' + str(resume) + '.pt')
