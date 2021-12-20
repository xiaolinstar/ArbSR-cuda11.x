import torch
import data
import loss
import model
import trainer
import utils
from option import option


def get_trainer(cp, params):
    torch.manual_seed(params.seed)

    # use PyTorch to train, what are needed:
    # my_loader
    # my_model
    # optimizer
    my_loader = data.Data(params)
    my_model = model.Model(params, cp)
    my_loss = loss.Loss(params, cp)
    return trainer.Trainer(params, my_loader, my_model, my_loss, cp)


if __name__ == '__main__':
    args = option.get_parse_from_json()
    checkpoint = utils.utility.CheckPoint(args)
    t = get_trainer(checkpoint, args)

    while not t.terminate():
        if args.test_only:
            t.test()
        else:
            t.train()
    checkpoint.done()
