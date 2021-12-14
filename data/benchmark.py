import os
from data import multiscalesrdata as SRdata


class Benchmark(SRdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        self.a_path = os.path.join(dir_data, 'benchmark', self.name)
        self.dir_hr = os.path.join(self.a_path, 'HR')
        self.dir_lr = os.path.join(self.a_path, 'LR_bicubic')
        self.ext = ('.png', '.png')
