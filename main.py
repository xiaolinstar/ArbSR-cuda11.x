import torch
import loss
import data
import model
import trainer
import utils.utility
from option import option

if __name__ == '__main__':
    args = option.get_parse_from_json()
    torch.manual_seed(args.seed)

    # checkpoint里面写的是什么我现在还不清楚
    checkpoint = utils.utility.checkpoint(args)

    # use PyTorch to train, what are needed:
    # loader
    # model
    # optimizer

    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = trainer.Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()

    checkpoint.done()
