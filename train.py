#!/usr/bin/env python
import optuna

try:
    import subprocess

    from google.colab import drive

    subprocess.run(["pip", "install", "torchmetrics"])
    base_dir = "/content/drive/MyDrive/Colab_Notebooks/Crack_Detection"
    drive.mount("/content/drive")
    LOCAL = False
except ImportError:
    base_dir = "."
    LOCAL = True

import datetime
import json
import os
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
from torchmetrics.clustering import RandScore
from torchvision.transforms import ElasticTransform

seed = 7
random.seed(seed)
torch.manual_seed(seed)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, init_features=64, dropout_p=0.5):
        # TODO adjust dropout probability
        super().__init__()
        features = init_features
        kernel_size = 3
        self.encoder1 = block(in_channels, features, kernel_size, "enc1")
        self.pool = nn.MaxPool2d(stride=2, kernel_size=2)
        self.encoder2 = block(features, features * 2, kernel_size, "enc2")
        self.encoder3 = block(features * 2, features * 4, kernel_size, "enc3")
        self.encoder4 = block(features * 4, features * 8, kernel_size, "enc4")
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.bottleneck_encoder = block(
            features * 8, features * 16, kernel_size, "bottleneck_enc"
        )
        self.up_conv1 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder1 = block(features * 16, features * 8, kernel_size, "dec1")
        self.up_conv2 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder2 = block(features * 8, features * 4, kernel_size, "dec2")
        self.up_conv3 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder3 = block(features * 4, features * 2, kernel_size, "dec3")
        self.up_conv4 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder4 = block(features * 2, features, kernel_size, "dec4")
        self.out_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        drop = self.dropout(enc4)
        bottleneck = self.bottleneck_encoder(self.pool(drop))

        up_conv1 = self.up_conv1(bottleneck)
        crop_bot, crop_top = crop_size(enc4, up_conv1)
        dec1 = self.decoder1(
            torch.cat(
                (enc4[:, :, crop_bot:crop_top, crop_bot:crop_top], up_conv1), dim=1
            )
        )

        up_conv2 = self.up_conv2(dec1)
        crop_bot, crop_top = crop_size(enc3, up_conv2)
        dec2 = self.decoder2(
            torch.cat(
                (enc3[:, :, crop_bot:crop_top, crop_bot:crop_top], up_conv2), dim=1
            )
        )

        up_conv3 = self.up_conv3(dec2)
        crop_bot, crop_top = crop_size(enc2, up_conv3)
        dec3 = self.decoder3(
            torch.cat(
                (enc2[:, :, crop_bot:crop_top, crop_bot:crop_top], up_conv3), dim=1
            )
        )

        up_conv4 = self.up_conv4(dec3)
        crop_bot, crop_top = crop_size(enc1, up_conv4)
        dec4 = self.decoder4(
            torch.cat(
                (enc1[:, :, crop_bot:crop_top, crop_bot:crop_top], up_conv4), dim=1
            )
        )

        output = self.out_conv(dec4)
        return output


def crop_size(encoder, up_conv) -> tuple:
    """Return crop size of encoder's feature maps so it fits up-conv's shape"""
    x = encoder.shape[2]
    y = up_conv.shape[2]
    return (x - y) // 2, (x + y) // 2


def block(
    in_channels: int, features: int, kernel_size: int, name: str
) -> nn.Sequential:
    return nn.Sequential(
        OrderedDict(
            [
                (f"{name}_conv1", nn.Conv2d(in_channels, features, kernel_size)),
                (f"{name}_relu1", nn.ReLU(inplace=True)),
                (f"{name}_conv2", nn.Conv2d(features, features, kernel_size)),
                (f"{name}_relu2", nn.ReLU(inplace=True)),
            ]
        )
    )


class SegmentationDataset(Dataset):
    def __init__(
        self, image_dir: str, mask_dir: str, image_names: list, transform=None
    ):
        self.transform = transform
        self.image_data = []
        self.mask_data = []
        for image_name in image_names:
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name)
            image = Image.open(image_path).convert("L")
            mask = Image.open(mask_path).convert("L")
            self.image_data.append(image)
            self.mask_data.append(mask)

    def __len__(self) -> int:
        return len(self.image_data)

    def __getitem__(self, idx: int) -> tuple:
        image = self.image_data[idx]
        mask = self.mask_data[idx]
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask


class ImageMaskTransform:
    def __init__(
        self,
        flip_prob=0.5,
        brightness_prob=0.5,
        rotate_prob=0.5,
        translate_prob=0.5,
        elastic_prob=0.5,
        elastic_alpha=50.0,
        elastic_sigma=5.0,
        # TODO adjust rotation degree, flip and rotation probabilities
        rotate_angle=30,
        translate_factor=0.1,
        min_brightness=0.1,
        max_brightness=2.0,
        train=True,
        image_size=512,
        input_size=572,
        mask_size=388,
        # TODO automatic rotate pad calc
        rotate_pad=186,
        elastic_pad=50,
    ):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.brightness_prob = brightness_prob
        self.translate_prob = translate_prob
        self.elastic_prob = elastic_prob
        self.translate_factor = translate_factor
        self.max_brightness = max_brightness
        self.min_brightness = min_brightness
        self.rotate_angle = rotate_angle
        self.train = train
        self.image_size = image_size
        self.input_size = input_size
        self.mask_size = mask_size
        self.default_pad = (input_size - image_size) // 2
        self.rotate_pad = rotate_pad
        self.elastic_pad = elastic_pad
        self.shift_pad = rotate_pad // 2
        self.elastic_transform = ElasticTransform(
            alpha=elastic_alpha, sigma=elastic_sigma
        )

    def align_inputs(self, image, mask):
        image = F.pad(image, padding=self.default_pad, padding_mode="reflect")
        mask = F.center_crop(mask, self.mask_size)
        return image, mask

    def __call__(self, image, mask):
        image = F.to_tensor(image)
        mask = F.to_tensor(mask).long()
        aligned = False
        if self.train:
            if random.random() < self.rotate_prob:
                image = F.pad(image, padding=self.rotate_pad, padding_mode="reflect")
                angle = random.uniform(-self.rotate_angle, self.rotate_angle)
                if random.random() < self.translate_prob:
                    mask = F.pad(mask, padding=self.shift_pad, padding_mode="reflect")
                    dx = random.uniform(-self.translate_factor, self.translate_factor)
                    dy = random.uniform(-self.translate_factor, self.translate_factor)
                    translate = (int(dx * self.image_size), int(dy * self.image_size))
                else:
                    translate = (0, 0)
                image = F.affine(image, angle, translate, scale=1.0, shear=0.0)
                mask = F.affine(mask, angle, translate, scale=1.0, shear=0.0)
                image = F.center_crop(image, self.input_size)
                mask = F.center_crop(mask, self.mask_size)
                aligned = True
            elif random.random() < self.elastic_prob:
                image = F.pad(image, padding=self.elastic_pad, padding_mode="reflect")
                mask = F.pad(mask, padding=self.elastic_pad, padding_mode="reflect")

                _, height, width = F.get_dimensions(image)
                displacement = self.elastic_transform.get_params(
                    self.elastic_transform.alpha,
                    self.elastic_transform.sigma,
                    [height, width],
                )

                interpolation = self.elastic_transform.interpolation
                fill = self.elastic_transform.fill
                image = F.elastic_transform(image, displacement, interpolation, fill)
                image = F.center_crop(image, self.input_size)

                mask = F.elastic_transform(mask, displacement, interpolation, fill)
                mask = F.center_crop(mask, self.mask_size)
                aligned = True

            if random.random() < self.flip_prob:
                image = F.hflip(image)
                mask = F.hflip(mask)

            if random.random() < self.flip_prob:
                image = F.vflip(image)
                mask = F.vflip(mask)

            if random.random() < self.brightness_prob:
                brightness_factor = random.uniform(
                    self.min_brightness, self.max_brightness
                )
                image = F.adjust_brightness(image, brightness_factor)

        if not self.train or not aligned:
            image, mask = self.align_inputs(image, mask)

        return image, mask


def cross_entropy_weighted(
    outputs, targets, device, w0=5, sigma=5, w1=1.0, vanilla=False
):
    if vanilla:
        return nn.functional.cross_entropy(outputs, targets)

    targets_bincount = torch.bincount(targets.flatten())
    weight_class = targets_bincount.sum() / targets_bincount * w1

    borders = find_boundaries(targets.cpu().numpy())
    dist = distance_transform_edt(~borders)
    dist = torch.tensor(dist).to(device)
    weight_borders = w0 * torch.exp(-2 * dist**2 / sigma**2)

    class_map = weight_class[targets]
    weight = weight_borders + class_map
    loss_map = nn.functional.cross_entropy(outputs, targets, reduction="none")
    return torch.mean(loss_map * weight)


def iou_loss(predictions, targets, eps=1e-6):
    intersection = torch.sum(predictions * targets)
    union = torch.sum(predictions + targets) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou


def train_fixed_hyperparams():
    (
        batch_size,
        train_image_dir,
        train_images,
        train_mask_dir,
        val_images,
        val_percent,
    ) = init_datasets()

    flip_prob = 0.1
    rotate_prob = 0.1
    elastic_prob = 0.11
    translate_prob = 0.1
    brightness_prob = 0.1
    train_dataloader, val_dataloader = init_data_loaders(
        batch_size,
        brightness_prob,
        elastic_prob,
        flip_prob,
        rotate_prob,
        train_image_dir,
        train_images,
        train_mask_dir,
        translate_prob,
        val_images,
    )

    lr = 1e-4
    num_epochs = 100
    min_save_epoch = 2
    loss_w0, loss_sigma = 5, 5
    loss_w1 = 1.0
    dropout_p = 0.2
    early_stop_patience = 30
    max_save_diff = 0.0005
    max_error_diff = 0.002
    overfit_patience = 5
    vanilla_loss = True
    use_adam = True
    use_cosine_scheduler = False
    min_lr = 1e-7
    # CosineAnnealingWarmRestarts
    t_0 = 1
    t_mult = 2
    # ReduceLROnPlateau
    lr_patience = 20
    lr_cooldown = 5
    lr_factor = 0.1

    fit(
        brightness_prob,
        dropout_p,
        early_stop_patience,
        elastic_prob,
        flip_prob,
        loss_sigma,
        loss_w0,
        loss_w1,
        lr,
        lr_cooldown,
        lr_factor,
        lr_patience,
        max_error_diff,
        max_save_diff,
        min_lr,
        min_save_epoch,
        num_epochs,
        overfit_patience,
        rotate_prob,
        t_0,
        t_mult,
        train_dataloader,
        translate_prob,
        use_adam,
        use_cosine_scheduler,
        val_dataloader,
        val_percent,
        vanilla_loss,
    )


def tune_hyperparams():
    def tuning_objective(trial: optuna.Trial):
        flip_prob = trial
        rotate_prob = 0.1
        elastic_prob = 0.11
        translate_prob = 0.1
        brightness_prob = 0.1
        train_dataloader, val_dataloader = init_data_loaders(
            batch_size,
            brightness_prob,
            elastic_prob,
            flip_prob,
            rotate_prob,
            train_image_dir,
            train_images,
            train_mask_dir,
            translate_prob,
            val_images,
        )

        lr = 1e-4
        num_epochs = 100
        min_save_epoch = 2
        loss_w0, loss_sigma = 5, 5
        loss_w1 = 1.0
        dropout_p = 0.2
        early_stop_patience = 30
        max_save_diff = 0.0005
        max_error_diff = 0.002
        overfit_patience = 5
        vanilla_loss = True
        use_adam = True
        use_cosine_scheduler = False
        min_lr = 1e-7
        # CosineAnnealingWarmRestarts
        t_0 = 1
        t_mult = 2
        # ReduceLROnPlateau
        lr_patience = 20
        lr_cooldown = 5
        lr_factor = 0.1

        fit(
            brightness_prob,
            dropout_p,
            early_stop_patience,
            elastic_prob,
            flip_prob,
            loss_sigma,
            loss_w0,
            loss_w1,
            lr,
            lr_cooldown,
            lr_factor,
            lr_patience,
            max_error_diff,
            max_save_diff,
            min_lr,
            min_save_epoch,
            num_epochs,
            overfit_patience,
            rotate_prob,
            t_0,
            t_mult,
            train_dataloader,
            translate_prob,
            use_adam,
            use_cosine_scheduler,
            val_dataloader,
            val_percent,
            vanilla_loss,
        )
    (
        batch_size,
        train_image_dir,
        train_images,
        train_mask_dir,
        val_images,
        val_percent,
    ) = init_datasets()

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=30, interval_steps=10
        ),
    )
    study.optimize(tuning_objective, n_trials=20)




def fit(
    brightness_prob,
    dropout_p,
    early_stop_patience,
    elastic_prob,
    flip_prob,
    loss_sigma,
    loss_w0,
    loss_w1,
    lr,
    lr_cooldown,
    lr_factor,
    lr_patience,
    max_error_diff,
    max_save_diff,
    min_lr,
    min_save_epoch,
    num_epochs,
    overfit_patience,
    rotate_prob,
    t_0,
    t_mult,
    train_dataloader,
    translate_prob,
    use_adam,
    use_cosine_scheduler,
    val_dataloader,
    val_percent,
    vanilla_loss,
):
    no_improve_epochs = 0
    overfit_epochs = 0
    best_rand_error = torch.tensor(float("inf"))
    best_epoch = -1
    best_weights = None
    model = UNet(dropout_p=dropout_p)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_weights_path = os.path.join(base_dir, "checkpoints/C1.pth")
    model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
    model.to(device)
    optimizer = (
        optim.Adam(model.parameters(), lr=lr)
        if use_adam
        else optim.SGD(model.parameters(), lr=lr)
    )
    scheduler = (
        optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t_0, T_mult=t_mult, eta_min=min_lr
        )
        if use_cosine_scheduler
        else optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=lr_factor,
            patience=lr_patience,
            cooldown=lr_cooldown,
            min_lr=min_lr,
        )
    )
    log_dir = os.path.join(
        base_dir, f"runs/run_{datetime.datetime.now().strftime('%m%d-%H%M%S')}"
    )
    writer = SummaryWriter(log_dir=log_dir)
    accuracy_metric_val = BinaryAccuracy().to(device)
    precision_metric_val = BinaryPrecision().to(device)
    recall_metric_val = BinaryRecall().to(device)
    rand_score_metric_val = RandScore().to(device)
    accuracy_metric_train = BinaryAccuracy().to(device)
    precision_metric_train = BinaryPrecision().to(device)
    recall_metric_train = BinaryRecall().to(device)
    rand_score_metric_train = RandScore().to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = cross_entropy_weighted(
                outputs,
                labels.squeeze(1),
                device,
                loss_w0,
                loss_sigma,
                loss_w1,
                vanilla=vanilla_loss,
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                labels = labels.squeeze(1)
                loss = cross_entropy_weighted(
                    outputs,
                    labels,
                    device,
                    loss_w0,
                    loss_sigma,
                    loss_w1,
                    vanilla=vanilla_loss,
                )
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                accuracy_metric_val.update(preds, labels)
                recall_metric_val.update(preds, labels)
                precision_metric_val.update(preds, labels)
                rand_score_metric_val.update(preds.view(-1), labels.view(-1))

            for images, labels in train_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                labels = labels.squeeze(1)
                preds = torch.argmax(outputs, dim=1)
                accuracy_metric_train.update(preds, labels)
                recall_metric_train.update(preds, labels)
                precision_metric_train.update(preds, labels)
                rand_score_metric_train.update(preds.view(-1), labels.view(-1))

        scalars = {
            "Loss/train": train_loss / len(train_dataloader),
            "RandError/train": 1 - rand_score_metric_train.compute(),
            "PixelError/train": 1 - accuracy_metric_train.compute(),
            "Recall/train": recall_metric_train.compute(),
            "Precision/train": precision_metric_train.compute(),
            "Loss/val": val_loss / len(val_dataloader),
            "RandError/val": 1 - rand_score_metric_val.compute(),
            "PixelError/val": 1 - accuracy_metric_val.compute(),
            "Recall/val": recall_metric_val.compute(),
            "Precision/val": precision_metric_val.compute(),
        }

        val_rand_error = scalars["RandError/val"]
        train_rand_error = scalars["RandError/train"]
        if use_cosine_scheduler:
            scheduler.step()
        else:
            scheduler.step(val_rand_error)

        for key, value in scalars.items():
            writer.add_scalar(key, value, epoch)
        writer.add_scalar("LearningRate/train", scheduler.get_last_lr()[0], epoch)

        print(f"Epoch {epoch + 1}/{num_epochs}, ", end="")
        print(", ".join([f"{key}: {value:.4f}" for key, value in scalars.items()]))

        accuracy_metric_val.reset()
        recall_metric_val.reset()
        precision_metric_val.reset()
        rand_score_metric_val.reset()

        accuracy_metric_train.reset()
        recall_metric_train.reset()
        precision_metric_train.reset()
        rand_score_metric_train.reset()

        val_train_diff = val_rand_error - train_rand_error
        if val_rand_error < best_rand_error:
            no_improve_epochs = 0
            best_rand_error = val_rand_error
            best_epoch = epoch + 1

            if best_epoch > min_save_epoch and val_train_diff < max_save_diff:
                best_weights = {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                }
                print(f"Saved the weights in RAM for the epoch {best_epoch}")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stop_patience:
            print(f"Early stop, best val rand error: {best_rand_error:.4f}")
            break

        if val_train_diff > max_error_diff:
            overfit_epochs += 1
            print(
                f"Overfitting detected, val rand error is greater by: {val_train_diff:.4f}"
            )

        if overfit_epochs >= overfit_patience:
            print("Early stop, overfit")
            break
    hparams_dict = dict(
        lr=lr,
        num_epochs=num_epochs,
        min_save_epoch=min_save_epoch,
        loss_w0=loss_w0,
        loss_sigma=loss_sigma,
        loss_w1=loss_w1,
        dropout_p=dropout_p,
        early_stop_patience=early_stop_patience,
        vanilla_loss=vanilla_loss,
        use_adam=use_adam,
        use_cosine_scheduler=use_cosine_scheduler,
        min_lr=min_lr,
        T_0=t_0,
        T_mult=t_mult,
        lr_patience=lr_patience,
        lr_cooldown=lr_cooldown,
        flip_prob=flip_prob,
        rotate_prob=rotate_prob,
        elastic_prob=elastic_prob,
        translate_prob=translate_prob,
        brightness_prob=brightness_prob,
        val_percent=val_percent,
    )
    with open(os.path.join(log_dir, "hparams.json"), "w") as f:
        json.dump(hparams_dict, f)
    writer.add_hparams(
        hparams_dict,
        {"hparam/rand_error": best_rand_error, "hparam/best_epoch": best_epoch},
    )
    writer.close()
    if best_weights:
        torch.save(best_weights, os.path.join(base_dir, "checkpoint.pth"))
        print("Saved the best weights after the training")


def init_data_loaders(
    batch_size,
    brightness_prob,
    elastic_prob,
    flip_prob,
    rotate_prob,
    train_image_dir,
    train_images,
    train_mask_dir,
    translate_prob,
    val_images,
):
    train_transform = ImageMaskTransform(
        flip_prob=flip_prob,
        rotate_prob=rotate_prob,
        elastic_prob=elastic_prob,
        translate_prob=translate_prob,
        brightness_prob=brightness_prob,
        elastic_alpha=200.0,
        elastic_sigma=7.0,
        rotate_angle=30,
        translate_factor=0.1,
        min_brightness=0.1,
        max_brightness=1.7,
    )
    train_dataset = SegmentationDataset(
        train_image_dir, train_mask_dir, train_images, transform=train_transform
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_transform = ImageMaskTransform(train=False)
    val_dataset = SegmentationDataset(
        train_image_dir, train_mask_dir, val_images, transform=val_transform
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader


def init_datasets():
    train_image_dir = os.path.join(base_dir, "isbi_2012_challenge/train/imgs")
    train_mask_dir = os.path.join(base_dir, "isbi_2012_challenge/train/labels")
    batch_size = 9
    val_percent = 0.2
    if LOCAL:
        batch_size = 1
    all_images = os.listdir(train_image_dir)
    val_size = int(val_percent * len(all_images))
    random.shuffle(all_images)
    val_images = all_images[:val_size]
    train_images = all_images[val_size:]
    return (
        batch_size,
        train_image_dir,
        train_images,
        train_mask_dir,
        val_images,
        val_percent,
    )
