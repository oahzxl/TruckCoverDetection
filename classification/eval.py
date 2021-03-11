import torch
import torch.utils.data
from utils.utils import batch_augment, get_metrics


def evaluate(data_loader, model, criterion, args, idx2class):

    with torch.no_grad():

        model.eval()
        total_loss = 0

        for data in data_loader:
            image = data["image"].cuda()
            label = data["label"].cuda()

            if args.model == "wsdan":
                y_pred_raw, feature_matrix, attention_map = model(image)
                crop_images = batch_augment(image, attention_map[:, :1, :, :],
                                            mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
                y_pred_crop, _, _ = model(crop_images)
                loss = (criterion(y_pred_raw, label) + criterion(y_pred_crop, label)) / 2
                predict = (y_pred_raw + y_pred_crop) / 2
            else:
                predict = model(image)
                loss = criterion(predict, label)

            _, predict = predict.max(1)

            for i in range(predict.shape[0]):
                if int(predict[i]) == 2:
                    predict[i] = 1
                if int(label[i]) == 2:
                    label[i] = 1

            if total_loss == 0:
                total_pred = predict
                total_label = label
            else:
                total_pred = torch.cat((total_pred, predict))
                total_label = torch.cat((total_label, label))
            total_loss += loss

        loss = total_loss / len(data_loader)
        metrics = get_metrics(total_pred, total_label, idx2class)
        metrics["loss"] = float(loss)

    return metrics
