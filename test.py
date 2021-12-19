import trainer_loader
import utils
from option import option

if __name__ == '__main__':
    args = option.get_parse_from_json()
    checkpoint = utils.utility.checkpoint(args)
    t = trainer_loader.get_trainer(checkpoint, args)

    while not t.terminate():
        t.test()
    checkpoint.done()
