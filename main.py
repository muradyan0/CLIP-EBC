import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import normalize, to_pil_image
from models import get_model
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
import sys
import json
from utils import resize_density_map, sliding_window_predict

def preprocess_image(image_path, input_size):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((input_size, input_size), Image.LANCZOS)
    return transforms.ToTensor()(image).unsqueeze(0)


def prepare_image_for_vit(image_path):
    # Load image
    image = Image.open(image_path)

    # Resize to 224x224
    image = Resize(224)(image)

    # Center crop to 224x224
    image = CenterCrop(224)(image)

    # Convert to tensor and normalize
    image_tensor = ToTensor()(image)
    image_tensor = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
        image_tensor
    )

    return image_tensor.unsqueeze(0)  # Add batch dimension


parent_dir = os.path.abspath('.')
sys.path.append(parent_dir)


def load_model(model_name, device):
    truncation = 4
    reduction = 8
    granularity = "fine"
    anchor_points = "average"
    dataset_name = "nwpu"

    prompt_type = "word"
    num_vpt = 32
    vpt_drop = 0.
    deep_vpt = True

    with open(os.path.join(parent_dir, "configs", f"reduction_{reduction}.json"), "r") as f:
        config = json.load(f)[str(truncation)][dataset_name]
    bins = config["bins"][granularity]
    anchor_points = config["anchor_points"][granularity]["average"] if anchor_points == "average" else config["anchor_points"][granularity]["middle"]
    bins = [(float(b[0]), float(b[1])) for b in bins]
    anchor_points = [float(p) for p in anchor_points]
    # return get_model(backbone=model_name, input_size=448, reduction=8).to(device)

    return get_model(
        backbone=model_name,
        input_size=448,
        reduction=reduction,
        anchor_points=anchor_points,
        bins=bins,

        prompt_type=prompt_type,
        num_vpt=num_vpt,
        vpt_drop=vpt_drop,
        deep_vpt=deep_vpt
    ).to(device)


def predict_count(image_path, model, device, input_img_size):
    # Preprocess the image
    image = preprocess_image(image_path, input_img_size).to(device)
    # image = prepare_image_for_vit(image_path)

    # Get predictions
    with torch.no_grad():
        output = model(image)
        pred_count = output.sum().item()

        print("predict", f"{pred_count:.0f}")

        image_height, image_width = image.shape[-2:]
        resized_pred_density = resize_density_map(output, (image_height, image_width)).cpu()
        resized_pred_density = resized_pred_density.squeeze().numpy()

        image = to_pil_image(image.squeeze(0))
        plt.imshow(image)
        plt.imshow(resized_pred_density, cmap="jet", alpha=0.8)
        plt.axis("off")
        plt.title(f"Колличество людей: {pred_count:.0f}")

        plt.savefig(f"./output.png", transparent=True, bbox_inches="tight", pad_inches=0)

        return


weight_path = "/root/CLIP-EBC/best_mae.pth"
# weight_path = "/root/CLIP-EBC/vit_best_mae_0.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = load_model("clip_vit_b_16", device)
model = load_model("clip_resnet50", device)
model.load_state_dict(torch.load(weight_path, map_location=device))
model.eval()
# image = "/root/CLIP-EBC/data/ShanghaiTech/part_A/train_data/images/IMG_1.jpg"
image = "/root/CLIP-EBC/test.jpg"

# image = "/root/CLIP-EBC/data/ShanghaiTech/part_A/test_data/images/IMG_4.jpg"

predict_count(image, model, device, 224)
