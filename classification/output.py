import os
import shutil

import torch
import torch.utils.data
from tqdm import tqdm

from utils.utils import batch_augment
from utils.utils import load_model


def output(data_loader, model, ckp_path, args):

    with torch.no_grad():

        model = load_model(model, ckp_path)
        model.eval()
        pic_dic = {}

        for data in data_loader:
            image = data["image"].cuda()
            name = data["name"][0][:-6]

            if args.model == "wsdan":
                y_pred_raw, feature_matrix, attention_map = model(image)
                with torch.no_grad():
                    crop_images = batch_augment(image, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6),
                                                padding_ratio=0.1)
                y_pred_crop, _, _ = model(crop_images)
                predict = (y_pred_raw + y_pred_crop) / 2
            else:
                predict = model(image)

            _, predict = predict.max(1)

            if name in pic_dic:
                pic_dic[name] = max(pic_dic[name], int(predict))
            else:
                pic_dic[name] = int(predict)
            
    origin_path = "../data/demo"
    target_path = "../data/output"
    legal_path = os.path.join(target_path, "legal")
    illegal_path = os.path.join(target_path, "illegal")
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.mkdir(target_path)
    os.mkdir(legal_path)
    os.mkdir(illegal_path)

    for k, v in pic_dic.items():
        k = k + ".jpg"
        if v == 0:
            shutil.copy(os.path.join(origin_path, k), os.path.join(legal_path))
            print(os.path.join(legal_path, k))
        else:
            shutil.copy(os.path.join(origin_path, k), os.path.join(illegal_path))
            print(os.path.join(illegal_path, k))

    print("Done.")
    return 0
