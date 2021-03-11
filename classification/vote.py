import torch
import torch.utils.data

from utils.utils import batch_augment, get_metrics
from utils.utils import load_model


def vote(data_loader, model, ckp_path, args, idx2class):

    with torch.no_grad():

        model = load_model(model, ckp_path)
        model.eval()
        pic_dic = {}

        for data in data_loader:
            image = data["image"].cuda()
            label = data["label"].cuda()
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

            for i in range(predict.shape[0]):
                if int(predict[i]) == 2:
                    predict[i] = 1
                if int(label[i]) == 2:
                    label[i] = 1

            if 'total_pred' not in locals():
                total_pred = predict
                total_label = label
            else:
                total_pred = torch.cat((total_pred, predict))
                total_label = torch.cat((total_label, label))

            if name in pic_dic:
                pic_dic[name][0] = max(pic_dic[name][0], int(predict))
                pic_dic[name][1] = max(pic_dic[name][1], int(label))
            else:
                pic_dic[name] = [int(predict), int(label)]

        pred = torch.zeros(len(pic_dic))
        label = torch.zeros(len(pic_dic))
        for i, v in enumerate(pic_dic.values()):
            pred[i] = v[0]
            label[i] = v[1]

        print("[normal]")
        get_metrics(total_pred, total_label, idx2class)

        print("[vote]")
        get_metrics(pred, label, idx2class)

    print("Done.")
    return 0
