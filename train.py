import os
import glob
import torch
import numpy as np


import albumentations
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import dataset
import utils

from model import CaptchaModel


from torch import nn


def run_training():
    if not os.path.exists(config.RESULTS):
        os.mkdir(config.RESULTS)
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.png"))
    targets_orig = [x.split("/")[-1][:-4] for x in image_files]
    targets = [[c for c in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]

    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)
    targets_enc = [lbl_enc.transform(x) for x in targets]
    targets_enc = np.array(targets_enc)
    targets_enc = targets_enc + 1

    (
        train_imgs,
        test_imgs,
        train_targets,
        test_targets,
        _,
        test_targets_orig,
    ) = model_selection.train_test_split(
        image_files, targets_enc, targets_orig, test_size=0.1, random_state=42
    )

    train_dataset = dataset.ClassificationDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
    )
    test_dataset = dataset.ClassificationDataset(
        image_paths=test_imgs,
        targets=test_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    model = CaptchaModel(num_chars=len(lbl_enc.classes_))
    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )

    epoch_list = []
    accuracy_list = []
    train_loss_list = []
    test_loss_list = []

    for epoch in range(config.EPOCHS):
        train_loss = utils.train_fn(model, train_loader, optimizer)

        valid_preds, test_loss = utils.eval_fn(model, test_loader)
        valid_captcha_preds = []
        for vp in valid_preds:
            current_preds = utils.decode_predictions(vp, lbl_enc)
            valid_captcha_preds.extend(current_preds)
        combined = list(zip(test_targets_orig, valid_captcha_preds))
        print(combined[:10])
        test_dup_rem = [utils.remove_duplicates(c) for c in test_targets_orig]
        accuracy = metrics.accuracy_score(test_dup_rem, valid_captcha_preds)
        print(
            f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={test_loss} Accuracy={accuracy}"
        )
        scheduler.step(test_loss)
        epoch_list.append(epoch)
        accuracy_list.append(accuracy)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        utils.plot_performance(epoch_list,accuracy_list,train_loss_list,test_loss_list,config.RESULTS)
        if epoch % 25 == 0:
            torch.save(model, config.RESULTS + "/model_" + str(epoch))
            torch.save(model.state_dict(), config.RESULTS + "/model_state _dict_" + str(epoch))
            print("Model is saved")

      



if __name__ == "__main__":
    run_training()
