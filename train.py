#!/usr/bin/env python
import datetime
import json
import math
import os
import pickle
import random
from collections import OrderedDict
from pathlib import Path

import click
import optuna
import torch
import torchvision.transforms.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
from torchmetrics.clustering import RandScore
from torchvision.transforms import ElasticTransform, PILToTensor

import gdrive

seed = 7
random.seed(seed)
torch.manual_seed(seed)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        init_features=64,
        dropout_p=0.5,
        padding=False,
    ):
        # TODO adjust dropout probability
        super().__init__()
        features = init_features
        kernel_size = 3
        self.encoder1 = block(in_channels, features, kernel_size, "enc1", padding)
        self.pool = nn.MaxPool2d(stride=2, kernel_size=2)
        self.encoder2 = block(features, features * 2, kernel_size, "enc2", padding)
        self.encoder3 = block(features * 2, features * 4, kernel_size, "enc3", padding)
        self.encoder4 = block(features * 4, features * 8, kernel_size, "enc4", padding)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.bottleneck_encoder = block(
            features * 8,
            features * 16,
            kernel_size,
            "bottleneck_enc",
            padding,
        )
        self.up_conv1 = nn.ConvTranspose2d(
            features * 16,
            features * 8,
            kernel_size=2,
            stride=2,
        )
        self.decoder1 = block(features * 16, features * 8, kernel_size, "dec1", padding)
        self.up_conv2 = nn.ConvTranspose2d(
            features * 8,
            features * 4,
            kernel_size=2,
            stride=2,
        )
        self.decoder2 = block(features * 8, features * 4, kernel_size, "dec2", padding)
        self.up_conv3 = nn.ConvTranspose2d(
            features * 4,
            features * 2,
            kernel_size=2,
            stride=2,
        )
        self.decoder3 = block(features * 4, features * 2, kernel_size, "dec3", padding)
        self.up_conv4 = nn.ConvTranspose2d(
            features * 2,
            features,
            kernel_size=2,
            stride=2,
        )
        self.decoder4 = block(features * 2, features, kernel_size, "dec4", padding)
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
                (enc4[:, :, crop_bot:crop_top, crop_bot:crop_top], up_conv1),
                dim=1,
            ),
        )

        up_conv2 = self.up_conv2(dec1)
        crop_bot, crop_top = crop_size(enc3, up_conv2)
        dec2 = self.decoder2(
            torch.cat(
                (enc3[:, :, crop_bot:crop_top, crop_bot:crop_top], up_conv2),
                dim=1,
            ),
        )

        up_conv3 = self.up_conv3(dec2)
        crop_bot, crop_top = crop_size(enc2, up_conv3)
        dec3 = self.decoder3(
            torch.cat(
                (enc2[:, :, crop_bot:crop_top, crop_bot:crop_top], up_conv3),
                dim=1,
            ),
        )

        up_conv4 = self.up_conv4(dec3)
        crop_bot, crop_top = crop_size(enc1, up_conv4)
        dec4 = self.decoder4(
            torch.cat(
                (enc1[:, :, crop_bot:crop_top, crop_bot:crop_top], up_conv4),
                dim=1,
            ),
        )

        output = self.out_conv(dec4)
        return output


def crop_size(encoder, up_conv) -> tuple:
    """Return crop size of encoder's feature maps so it fits up-conv's shape"""
    x = encoder.shape[2]
    y = up_conv.shape[2]
    return (x - y) // 2, (x + y) // 2


def block(
    in_channels: int,
    features: int,
    kernel_size: int,
    name: str,
    use_padding: bool,
) -> nn.Sequential:
    padding = "same" if use_padding else 0
    return nn.Sequential(
        OrderedDict(
            [
                (
                    f"{name}_conv1",
                    nn.Conv2d(in_channels, features, kernel_size, padding=padding),
                ),
                (f"{name}_relu1", nn.ReLU(inplace=True)),
                (
                    f"{name}_conv2",
                    nn.Conv2d(features, features, kernel_size, padding=padding),
                ),
                (f"{name}_relu2", nn.ReLU(inplace=True)),
            ],
        ),
    )


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_names: list,
        transform=None,
        foreground_value=255,
    ):
        self.transform = transform
        self.image_data = []
        self.mask_data = []
        for image_name in image_names:
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name)
            image = Image.open(image_path).convert("L")
            mask = Image.open(mask_path).convert("L")
            image = F.to_tensor(image)
            mask = (PILToTensor()(mask) == foreground_value).long()
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
        mean=0.4924,
        std=0.1735,
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
            alpha=elastic_alpha,
            sigma=elastic_sigma,
        )
        self.mean = mean
        self.std = std

    def align_inputs(self, image, mask):
        if self.default_pad:
            image = F.pad(image, padding=self.default_pad, padding_mode="reflect")
            mask = F.center_crop(mask, self.mask_size)
        return image, mask

    def __call__(self, image, mask):
        aligned = False
        if self.train:
            if random.random() < self.rotate_prob:
                image, mask = self.affine(image, mask)
                aligned = True
            elif random.random() < self.elastic_prob:
                image, mask = self.elastic(image, mask)
                aligned = True

            if random.random() < self.flip_prob:
                image = F.hflip(image)
                mask = F.hflip(mask)

            if random.random() < self.flip_prob:
                image = F.vflip(image)
                mask = F.vflip(mask)

            if random.random() < self.brightness_prob:
                brightness_factor = random.uniform(
                    self.min_brightness,
                    self.max_brightness,
                )
                image = F.adjust_brightness(image, brightness_factor)

        if not self.train or not aligned:
            image, mask = self.align_inputs(image, mask)

        image = F.normalize(image, [self.mean], [self.std])
        return image, mask

    def elastic(self, image, mask):
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
        return image, mask

    def affine(self, image, mask):
        image = F.pad(image, padding=self.rotate_pad, padding_mode="reflect")
        if not self.default_pad:
            mask = F.pad(mask, padding=self.rotate_pad, padding_mode="reflect")
        angle = random.uniform(-self.rotate_angle, self.rotate_angle)
        if random.random() < self.translate_prob:
            mask, translate = self.translate(mask)
        else:
            translate = (0, 0)
        image = F.affine(image, angle, translate, scale=1.0, shear=0.0)
        mask = F.affine(mask, angle, translate, scale=1.0, shear=0.0)
        image = F.center_crop(image, self.input_size)
        mask = F.center_crop(mask, self.mask_size)
        return image, mask

    def translate(self, mask):
        if self.default_pad:
            mask = F.pad(mask, padding=self.shift_pad, padding_mode="reflect")
        dx = random.uniform(-self.translate_factor, self.translate_factor)
        dy = random.uniform(-self.translate_factor, self.translate_factor)
        translate = (int(dx * self.image_size), int(dy * self.image_size))
        return mask, translate


def cross_entropy_weighted(
    outputs,
    targets,
    device,
    w0=5,
    sigma=5,
    w1=1.0,
    vanilla=False,
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


def train_fixed_hyperparams(
    base_dir: str,
    local: bool,
    params: dict,
    num_epochs: int,
    min_save_epoch: int,
    early_stop_patience: int,
    num_workers: int,
):
    (train_image_dir, train_images, train_mask_dir, val_images, val_percent) = (
        init_datasets(base_dir)
    )

    flip_prob = params["flip_prob"]
    rotate_prob = params["rotate_prob"]
    elastic_prob = params["elastic_prob"]
    translate_prob = params["translate_prob"]
    brightness_prob = params["brightness_prob"]
    batch_size = params["batch_size"]
    lr = params["lr"]
    dropout_p = params["dropout_p"]
    vanilla_loss = params["vanilla_loss"]
    use_adam = params["use_adam"]
    use_cosine_scheduler = params["use_cosine_scheduler"]
    min_lr = params["min_lr"]
    t_0 = params.get("t_0", 1)
    t_mult = params.get("t_mult", 2)
    lr_patience = params.get("lr_patience", 20)
    lr_cooldown = params.get("lr_cooldown", 5)
    lr_factor = params.get("lr_factor", 0.1)
    momentum = params.get("momentum", 0.99)
    loss_w0 = params.get("loss_w0", 5)
    loss_sigma = params.get("loss_sigma", 5)
    loss_w1 = params.get("loss_w1", 1.0)
    padding = params.get("padding", False)
    mask_size = 512 if padding else 388
    input_size = 512 if padding else 572

    if local:
        batch_size = 1

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
        num_workers,
        input_size,
        mask_size,
    )
    fit(
        brightness_prob,
        padding,
        dropout_p,
        early_stop_patience,
        elastic_prob,
        flip_prob,
        loss_sigma,
        loss_w0,
        loss_w1,
        lr,
        momentum,
        lr_cooldown,
        lr_factor,
        lr_patience,
        min_lr,
        min_save_epoch,
        num_epochs,
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
        base_dir,
        local,
    )


def tune_hyperparams(base_dir: str, local: bool):
    def tuning_objective(trial: optuna.Trial):
        flip_prob = trial.suggest_float("flip_prob", 0.001, 0.6)
        rotate_prob = trial.suggest_float("rotate_prob", 0.001, 0.6)
        elastic_prob = trial.suggest_float("elastic_prob", 0.001, 0.6)
        translate_prob = trial.suggest_float("translate_prob", 0.001, 0.6)
        brightness_prob = trial.suggest_float("brightness_prob", 0.001, 0.6)
        batch_size = trial.suggest_int("batch_size", 8, 11)
        padding = False
        if local:
            batch_size = 1
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

        dropout_p = trial.suggest_float("dropout_p", 0.01, 0.6)
        bool_choices = (True, False)
        vanilla_loss = trial.suggest_categorical("vanilla_loss", bool_choices)
        if vanilla_loss:
            loss_w0, loss_sigma, loss_w1 = 0.1, 0.1, 0.1
        else:
            loss_w0 = trial.suggest_float("loss_w0", 0.1, 10.0)
            loss_sigma = trial.suggest_float("loss_sigma", 0.1, 10.0)
            loss_w1 = trial.suggest_float("loss_w1", 0.1, 10.0)

        use_adam = trial.suggest_categorical("use_adam", bool_choices)
        if use_adam:
            lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
            momentum = 0.99
        else:
            lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
            momentum = trial.suggest_float("momentum", 0.1, 0.995)
        use_cosine_scheduler = trial.suggest_categorical(
            "use_cosine_scheduler",
            bool_choices,
        )
        min_lr = trial.suggest_float("min_lr", 1e-8, 1e-7, log=True)

        if use_cosine_scheduler:
            configs = [(1, 2), (10, 2), (50, 1), (100, 1)]
            config_index = trial.suggest_categorical(
                "cosine_scheduler_config",
                tuple(range(len(configs))),
            )
            t_0, t_mult = configs[config_index]
            lr_patience = 20
            lr_cooldown = 5
            lr_factor = 0.1
        else:
            t_0 = 1
            t_mult = 2
            lr_patience = trial.suggest_int("lr_patience", 16, 30, step=2)
            lr_cooldown = trial.suggest_int("lr_cooldown", 0, 5)
            lr_factor = trial.suggest_float("lr_factor", 0.05, 0.5, log=True)

        return fit(
            brightness_prob,
            padding,
            dropout_p,
            early_stop_patience,
            elastic_prob,
            flip_prob,
            loss_sigma,
            loss_w0,
            loss_w1,
            lr,
            momentum,
            lr_cooldown,
            lr_factor,
            lr_patience,
            min_lr,
            min_save_epoch,
            num_epochs,
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
            base_dir,
            local,
            trial,
        )

    (train_image_dir, train_images, train_mask_dir, val_images, val_percent) = (
        init_datasets(base_dir)
    )

    num_epochs = 120
    early_stop_patience = 40
    min_save_epoch = num_epochs
    save_study_interval = 10

    data_dir = Path(base_dir, "Data")
    data_dir.mkdir(exist_ok=True)
    storage_file = data_dir / "seg-study.db"
    storage = f"sqlite:///{storage_file}"
    study_name = f"study-{datetime.datetime.now().strftime('%m%d-%H%M%S')}"
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=10,
            max_resource=num_epochs,
            reduction_factor=2,
        ),
    )

    def save_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if not trial.number:
            return
        if trial.number % save_study_interval == 0:
            torch.cuda.empty_cache()
            save_study(data_dir, study, f"{study_name}-trial-{trial.number}")

    try:
        study.optimize(tuning_objective, n_trials=100, callbacks=[save_callback])
    except Exception as e:
        print("Error during optimization: ", e)
    finally:
        save_study(data_dir, study, study_name)

        container_id = os.environ.get("CONTAINER_ID")
        os.system(f"vastai stop instance {container_id}")


def save_study(data_dir, study, study_name):
    studies_dir = data_dir / "studies"
    studies_dir.mkdir(exist_ok=True)
    with open(studies_dir / f"{study_name}.pkl", "wb") as study_file:
        pickle.dump({"pruner": study.pruner, "sampler": study.sampler}, study_file)

    gdrive.upload_experiment(study_name)


def fit(
    brightness_prob,
    padding,
    dropout_p,
    early_stop_patience,
    elastic_prob,
    flip_prob,
    loss_sigma,
    loss_w0,
    loss_w1,
    lr,
    momentum,
    lr_cooldown,
    lr_factor,
    lr_patience,
    min_lr,
    min_save_epoch,
    num_epochs,
    rotate_prob,
    t_0,
    t_mult,
    train_dataloader,
    translate_prob,
    use_adam,
    use_cosine_scheduler,
    val_dataloader,
    val_percent,
    use_vanilla_loss,
    base_dir: str,
    local: bool,
    trial=None,
):
    no_improve_epochs = 0
    best_rand_error = torch.tensor(float("inf"))
    best_epoch = -1
    best_weights = None
    model = UNet(dropout_p=dropout_p, padding=padding)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = (
        optim.Adam(model.parameters(), lr=lr)
        if use_adam
        else optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    )
    scheduler = (
        optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t_0,
            T_mult=t_mult,
            eta_min=min_lr,
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
        base_dir,
        f"runs/run_{datetime.datetime.now().strftime('%m%d-%H%M%S')}",
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
    is_loss_invalid = False
    try:
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
                    vanilla=use_vanilla_loss,
                )
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                if math.isnan(train_loss) or math.isinf(train_loss):
                    is_loss_invalid = True
                    break
                if local:
                    break

            if is_loss_invalid:
                break
            model.eval()
            val_loss, vanilla_loss_val = 0.0, 0.0
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
                        vanilla=use_vanilla_loss,
                    )
                    loss_vanilla = cross_entropy_weighted(
                        outputs,
                        labels,
                        device,
                        loss_w0,
                        loss_sigma,
                        loss_w1,
                        vanilla=True,
                    )
                    vanilla_loss_val += loss_vanilla.item()
                    val_loss += loss.item()
                    if math.isnan(val_loss) or math.isinf(val_loss):
                        is_loss_invalid = True
                        break

                    preds = torch.argmax(outputs, dim=1)
                    accuracy_metric_val.update(preds, labels)
                    recall_metric_val.update(preds, labels)
                    precision_metric_val.update(preds, labels)
                    rand_score_metric_val.update(preds.view(-1), labels.view(-1))
                    if local:
                        break

                if is_loss_invalid:
                    break

                for images, labels in train_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    labels = labels.squeeze(1)
                    preds = torch.argmax(outputs, dim=1)
                    accuracy_metric_train.update(preds, labels)
                    recall_metric_train.update(preds, labels)
                    precision_metric_train.update(preds, labels)
                    rand_score_metric_train.update(preds.view(-1), labels.view(-1))
                    if local:
                        break

            scalars = {
                "Loss/train": train_loss / len(train_dataloader),
                "RandError/train": 1 - rand_score_metric_train.compute(),
                "PixelError/train": 1 - accuracy_metric_train.compute(),
                "Recall/train": recall_metric_train.compute(),
                "Precision/train": precision_metric_train.compute(),
                "Loss/val": val_loss / len(val_dataloader),
                "VanillaLoss/val": vanilla_loss_val / len(val_dataloader),
                "RandError/val": 1 - rand_score_metric_val.compute(),
                "PixelError/val": 1 - accuracy_metric_val.compute(),
                "Recall/val": recall_metric_val.compute(),
                "Precision/val": precision_metric_val.compute(),
            }

            val_rand_error = scalars["RandError/val"]
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

            if val_rand_error < best_rand_error:
                no_improve_epochs = 0
                best_rand_error = val_rand_error
                best_epoch = epoch + 1

                if best_epoch > min_save_epoch:
                    best_weights = {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    }
                    print(f"Saved the weights in RAM for the epoch {best_epoch}")
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= early_stop_patience:
                print(f"Early stop, best val rand error: {best_rand_error:.4f}")
                break

            if trial:
                trial.report(scalars["VanillaLoss/val"], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
    except KeyboardInterrupt:
        print("KeyboardInterrupt: stop training")
    except Exception as e:
        print("Exception occurred during train loop:", e)

    hparams_dict = dict(
        lr=lr,
        momentum=momentum,
        num_epochs=num_epochs,
        min_save_epoch=min_save_epoch,
        loss_w0=loss_w0,
        loss_sigma=loss_sigma,
        loss_w1=loss_w1,
        dropout_p=dropout_p,
        early_stop_patience=early_stop_patience,
        vanilla_loss=use_vanilla_loss,
        use_adam=use_adam,
        use_cosine_scheduler=use_cosine_scheduler,
        min_lr=min_lr,
        T_0=t_0,
        T_mult=t_mult,
        lr_patience=lr_patience,
        lr_cooldown=lr_cooldown,
        lr_factor=lr_factor,
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

    return best_rand_error


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
    num_workers=4,
    input_size=572,
    mask_size=388,
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
        max_brightness=1.6,
        input_size=input_size,
        mask_size=mask_size,
    )
    train_dataset = SegmentationDataset(
        train_image_dir,
        train_mask_dir,
        train_images,
        transform=train_transform,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_transform = ImageMaskTransform(
        train=False,
        input_size=input_size,
        mask_size=mask_size,
    )
    val_dataset = SegmentationDataset(
        train_image_dir,
        train_mask_dir,
        val_images,
        transform=val_transform,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_dataloader, val_dataloader


def init_datasets(base_dir: str) -> tuple[str, list, str, list, float]:
    train_image_dir = os.path.join(base_dir, "isbi_2012_challenge/train/imgs")
    train_mask_dir = os.path.join(base_dir, "isbi_2012_challenge/train/labels")
    val_percent = 0.2
    all_images = os.listdir(train_image_dir)
    val_size = int(val_percent * len(all_images))
    random.shuffle(all_images)
    val_images = all_images[:val_size]
    train_images = all_images[val_size:]
    return train_image_dir, train_images, train_mask_dir, val_images, val_percent


@click.command()
@click.option("--tune", is_flag=True, type=bool)
def main(tune):
    local = False
    base_dir = "."
    if tune:
        tune_hyperparams(base_dir, local)
    else:
        params = {
            "batch_size": 8,
            "brightness_prob": 0.3,
            "dropout_p": 0.5329,
            "elastic_prob": 0.03,
            "flip_prob": 0.3,
            "lr": 0.000363,
            "lr_cooldown": 40,
            "lr_factor": 0.7,
            "lr_patience": 45,
            "min_lr": 1e-7,
            "rotate_prob": 0.472,
            "translate_prob": 0.3,
            "loss_w0": 1.0,
            "loss_sigma": 5.0,
            "loss_w1": 0.0,
            "use_adam": True,
            "use_cosine_scheduler": False,
            "vanilla_loss": False,
            "padding": True,
        }
        num_epochs = 1200
        min_save_epoch = 50
        early_stop_patience = 200
        num_workers = 2
        best_error = train_fixed_hyperparams(
            base_dir,
            local,
            params,
            num_epochs,
            min_save_epoch,
            early_stop_patience,
            num_workers,
        )
        print(best_error)


if __name__ == "__main__":
    main()
