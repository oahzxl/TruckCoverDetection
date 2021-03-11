import os
import shutil

import torch
import torch.utils.data
from tqdm import tqdm

from utils.utils import batch_augment
from utils.utils import load_model


def vis(data_loader, model, ckp_path, args):

    with torch.no_grad():

        model = load_model(model, ckp_path)
        model.eval()
        pic_dic = {}

        outputs = []

        idx2class = {0: "empty", 1: "illegal", 2: "legal"}

        if not os.path.exists("./vis"):
            os.mkdir("./vis")
        for i in ["/empty", "/legal", "/illegal"]:
            if not os.path.exists("./vis" + i):
                os.mkdir("./vis" + i)
            for j in ["/empty", "/legal", "/illegal"]:
                if not os.path.exists("./vis" + i + j):
                    os.mkdir("./vis" + i + j)

        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):

            image = data["image"].cuda()
            name = data["name"]
            label = int(data["label"])

            if args.model == "wsdan":
                y_pred_raw, _, attention_map = model(image)
                with torch.no_grad():
                    crop_images = batch_augment(image, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6),
                                                padding_ratio=0.1)
                y_pred_crop, _, _ = model(crop_images)
                output = (y_pred_raw + y_pred_crop) / 2

            else:
                output = model(image)
            output = int(torch.argmax(output, dim=-1))

            save_path = "./vis/" + idx2class[label] + "/" + idx2class[output] + "/" + name[0]
            shutil.copy(os.path.join(args.data_path, "eval", name), save_path)

    print("Done.")
    return 0
