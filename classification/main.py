import argparse

import torch.utils.data
from resnest.torch import *

from models import *
from train import train
from utils import *
from vote import vote
from vis import vis
from output import output

class2idx = {"illegal": 0, "empty": 1, "legal": 2}
idx2class = {0: "illegal", 1: "legal"}


def init_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--mode', type=str, default="train", help='train/test')
    # parser.add_argument('--mode', type=str, default="output", help='train/test')
    parser.add_argument('--mode', type=str, default="vote", help='train/test')
    # parser.add_argument('--mode', type=str, default="vis", help='train/test')

    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default="Adam")

    # parser.add_argument('--sche', type=str, default="None")
    parser.add_argument('--sche', type=str, default="cos")
    # parser.add_argument('--sche', type=str, default="reduce")

    # parser.add_argument('--model', type=str, default="wsdan")
    # parser.add_argument('--model', type=str, default="hrnet")
    # parser.add_argument('--model', type=str, default="res_18")
    # parser.add_argument('--model', type=str, default="res_101")
    # parser.add_argument('--model', type=str, default="res_cbam")
    # parser.add_argument('--model', type=str, default="res_wsl")
    # parser.add_argument('--model', type=str, default="senet")
    # parser.add_argument('--model', type=str, default="efficient")
    parser.add_argument('--model', type=str, default="resnest")

    parser.add_argument('--factor', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--t0', type=int, default=1)
    parser.add_argument('--tm', type=int, default=2)

    parser.add_argument('--num-epochs', type=int, default=15)
    parser.add_argument('--print-interval', type=int, default=9999)
    parser.add_argument('--num-attentions', type=int, default=8)
    parser.add_argument('--h', type=int, default=500)
    parser.add_argument('--w', type=int, default=500)
    parser.add_argument('--beta', type=float, default=5e-2)
    parser.add_argument('--num-workers', type=int, default=16)

    # parser.add_argument('--data-path', type=str, default="../data/gt_crop1.2")
    parser.add_argument('--data-path', type=str, default="../data/gt_crop1")
    # parser.add_argument('--data-path', type=str, default="../data/cropb")

    parser.add_argument('--ckp-path', type=str, default="./log/model.tar")
    parser.add_argument('--save-path', type=str, default="./log/tmp/")
    if not os.path.exists(parser.parse_args().save_path):
        os.mkdir(parser.parse_args().save_path)

    if parser.parse_args().model == "hrnet":
        parser.add_argument('--cfg', help='experiment configure file name', type=str,
                            default=r"./models/hrnet/experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml")
        parser.add_argument('--modelDir', help='model directory', type=str, default='')
        parser.add_argument('--logDir', help='log directory', type=str, default='')
        parser.add_argument('--dataDir', help='data directory', type=str, default='')
        parser.add_argument('--testModel', help='testModel', type=str,
                            default=r"./data/pre-trained/hrnetv2_w48_imagenet_pretrained.pth")

    return parser.parse_args()


if __name__ == "__main__":

    # setup_seed(2020)
    args = init_args()
    print(args)
    print("Start loading data")

    if args.model == "wsdan":
        model = WSDAN(num_classes=len(class2idx), M=32, net='inception_mixed_6e', pretrained=True)
        # model = WSDAN(num_classes=len(class2idx), M=32, net='resnest', pretrained=True)
    elif args.model == "res_cbam":
        model = resnext101_32x8d(pretrained=True, progress=True)
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, len(class2idx)))
    elif args.model == "res_cbam_nd":
        model = resnext101_32x8d(pretrained=True, progress=True)
        model.fc = nn.Linear(2048, len(class2idx))
    elif args.model == "res_wsl":
        model = resnext101_32x8d_wsl(progress=True)
        # model = resnext101_32x16d_wsl(progress=True)
        model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(2048, len(class2idx)))
    elif args.model == "res_18":
        model = resnet18(pretrained=True, progress=True)
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, len(class2idx)))
    elif args.model == "res_101":
        model = resnet101(pretrained=True, progress=True)
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, len(class2idx)))
    elif args.model == "senet":
        model = se_resnext101(num_classes=len(class2idx))
    elif args.model == "efficient":
        model = efficientnet(size='b4', num_classes=len(class2idx))  # too big
    elif args.model == "resnest":
        model = resnest101(pretrained=True)  # 50, 101, 200, 269
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, len(class2idx)))
    elif args.model == "hrnet":
        from models.hrnet.lib.config import config
        from models.hrnet.lib.config import update_config
        from models.hrnet.lib.models.cls_hrnet import get_cls_net
        update_config(config, args)
        model = get_cls_net(config)
        model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(2048, len(class2idx)))
    else:
        raise ValueError
    model.cuda()

    if args.mode == "train":
        train_dataset = data_loader.build_dataset(args.data_path, (args.h, args.w), mode="train", class2idx=class2idx)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, shuffle=True, drop_last=True)
        eval_dataset = data_loader.build_dataset(args.data_path, (args.h, args.w), mode="eval", class2idx=class2idx)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size,
                                                  num_workers=args.num_workers, shuffle=True, drop_last=True)
        train(model, train_loader, eval_loader, args, idx2class)

    if args.mode == "output":
        output_dataset = data_loader.build_dataset(args.data_path, (args.h, args.w), mode="eval", class2idx=class2idx)
        output_loader = torch.utils.data.DataLoader(output_dataset, batch_size=1, shuffle=False)
        output(output_loader, model, args.ckp_path, args)

    if args.mode == "vis":
        vis_dataset = data_loader.build_dataset(args.data_path, (args.h, args.w), mode="eval", class2idx=class2idx)
        vis_loader = torch.utils.data.DataLoader(vis_dataset, batch_size=1, shuffle=False)
        vis(vis_loader, model, args.ckp_path, args)

    if args.mode == "vote":
        vote_dataset = data_loader.build_dataset(args.data_path, (args.h, args.w), mode="eval", class2idx=class2idx)
        vote_loader = torch.utils.data.DataLoader(vote_dataset, batch_size=1, shuffle=False)
        vote(vote_loader, model, args.ckp_path, args, idx2class)
