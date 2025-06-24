import click
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import torch
from train import UNet

from pathlib import Path


@click.command("Run model inference on an image")
@click.argument("model_weights", type=str)
@click.argument("img_path", type=str)
def main(img_path, model_weights):
    dropout_p = 0.5329
    mean = 0.4924
    std = 0.1735
    padding = True

    model = UNet(dropout_p=dropout_p, padding=padding)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()

    file = Path(img_path)
    img = Image.open(file).convert("L")
    images = F.to_tensor(img).unsqueeze(0)
    images = F.normalize(images, mean=[mean], std=[std])
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        predicted_mask = torch.argmax(outputs[0], dim=0).cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Input image")
    axs[0].axis("off")
    axs[1].imshow(img, cmap="gray")
    axs[1].imshow(predicted_mask, cmap="Purples", alpha=0.4)
    axs[1].set_title("Prediction")
    axs[1].axis("off")
    plt.savefig(f"demo/{file.name}")


if __name__ == "__main__":
    main()
