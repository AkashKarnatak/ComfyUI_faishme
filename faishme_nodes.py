import os
import numpy as np
import torch
import folder_paths as comfy_paths
import node_helpers
from PIL import Image, ImageOps


def load_image(img_path):
    img = node_helpers.pillow(Image.open, img_path)

    i = node_helpers.pillow(ImageOps.exif_transpose, img)

    if i.mode == "I":
        i = i.point(lambda i: i * (1 / 255))
    image = i.convert("RGB")

    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]

    return image


class ChooseFashionModel:
    def __init__(self):
        self.base_model_dir = os.path.join(comfy_paths.base_path, "faishme", "models")
        self.base_mask_dir = os.path.join(comfy_paths.base_path, "faishme", "masks")

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "ethnicity": (["asian", "black", "caucasian", "indian"],),
                "gender": (["female", "male"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "get_random_model"
    CATEGORY = "FaishmeNodes"

    def get_random_model(self, ethnicity, gender):
        model_dir = os.path.join(self.base_model_dir, gender, ethnicity)
        mask_dir = os.path.join(self.base_mask_dir, gender, ethnicity)

        model_images = sorted(os.listdir(model_dir))
        mask_images = sorted(os.listdir(mask_dir))
        idx = np.random.randint(len(model_images))

        model_img = load_image(model_images[idx])
        mask_img = load_image(mask_images[idx])

        mask = mask_img[:, :, :, 0]

        return (model_img, mask)


NODE_CLASS_MAPPINGS = {
    "Choose Fashion Model": ChooseFashionModel
}
