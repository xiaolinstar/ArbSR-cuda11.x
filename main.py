import torch

import argparse

import data
import model
import utils.utility
from option import option

if __name__ == '__main__':
    args = option.get_parse_from_json()
    torch.manual_seed(args.seed)
    checkpoint = utils.utility.checkpoint(args)
    loader = data.Data(args)
    model = model.Model(args, checkpoint)


