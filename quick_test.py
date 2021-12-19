import imageio
import numpy as np
import torch

import option.option
import utils.utility
from model.arbrcan import ArbRCAN


# 开始使用前请设置以下参数
# dir_img
# sr_size
# resume
if __name__ == '__main__':
    args = option.option.get_parse_from_json()

    device = None
    if args.n_GPUs > 0:
        device = 'cuda:0'
    else:
        device = 'cpu'

    my_model = ArbRCAN(args).to(device)
    print(args.resume)

    ckp = torch.load('experiment/ArbRCAN/model/model_'+str(args.resume)+'.pt', map_location=device)

    my_model.load_state_dict(ckp)

    my_model.eval()

    # load lr image
    lr = imageio.imread(args.dir_img)
    lr = np.array(lr)
    lr = torch.Tensor(lr).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)

    assert 1 < args.sr_size[0] / lr.size(2) <= 4
    assert 1 < args.sr_size[1] / lr.size(3) <= 4

    with torch.no_grad():
        scale_1 = args.sr_size[0] / lr.size(2)
        scale_2 = args.sr_size[1] / lr.size(3)
        my_model.set_scale(scale_1, scale_2)

        sr = my_model(lr)
        sr = utils.utility.quantize(sr, args.rgb_range)
        sr = sr.data.mul(255 / args.rgb_range)
        sr = sr[0, ...].permute(1, 2, 0).cpu().numpy()
        filename = 'experiment/quick_test/results/{}x{}'.format(int(args.sr_size[0]), int(args.sr_size[1]))
        imageio.imsave('{}.png'.format(filename), sr)

