import torch
import loss
import data
import model
import trainer
import utils.utility
from option import option


def get_trainer(checkpoint, args):
    torch.manual_seed(args.seed)

    # use PyTorch to train, what are needed:
    # my_loader
    # my_model
    # optimizer
    my_loader = data.Data(args)
    my_model = model.Model(args, checkpoint)
    my_loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = trainer.Trainer(args, my_loader, my_model, my_loss, checkpoint)
    return t
