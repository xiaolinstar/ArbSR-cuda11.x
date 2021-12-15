from importlib import import_module

from torch.utils.data import DataLoader


class Data:
    def __init__(self, args):
        self.loader_train = None

        if not args.test_only:
            # load the right dataset loader module
            # default: div2k.py
            # you can define your model and decide it's name
            module_train = import_module('data.' + args.data_train.lower())
            # load the dataset, args.data_train is the  dataset name
            # default: div2k, DIV2K
            train_set = getattr(module_train, args.data_train)(args)

            self.loader_train = DataLoader(dataset=train_set, batch_size=args.batch_size,
                                           shuffle=True, pin_memory=not args.cpu)

        # load the test dataset
        if args.data_test in ['Set5', 'Set14', 'B100', 'Manga109', 'Urban100']:
            module_test = import_module('data.benchmark')
            test_set = getattr(module_test, 'Benchmark')(args, name=args.data_test, train=False)
        else:
            module_test = import_module('data.' + args.data_test.lower())
            test_set = getattr(module_test, args.data_test)(args, train=False)

        self.loader_train = DataLoader(dataset=test_set, batch_size=args.batch_size,
                                       shuffle=True, pin_memory=not args.cpu)

