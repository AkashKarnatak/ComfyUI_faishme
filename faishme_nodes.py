import os
from glob import glob
import numpy as np
import torch
import folder_paths as comfy_paths
import node_helpers
from PIL import Image, ImageOps
from transformers import AutoModelForCausalLM, AutoTokenizer

import folder_paths


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy()[0], 0, 255).astype(np.uint8)
    )


# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def load_image(img_path):
    img = node_helpers.pillow(Image.open, img_path)

    i = node_helpers.pillow(ImageOps.exif_transpose, img)

    if i.mode == "I":
        i = i.point(lambda i: i * (1 / 255))
    image = i.convert("RGB")

    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]

    return image


def detect_device():
    """
    Detects the appropriate device to run on, and return the device and dtype.
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    else:
        return torch.device("cpu"), torch.float32


# from https://github.com/pythongosssss/ComfyUI-Custom-Scripts
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any = AnyType("*")


class LoadFashionModel:
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

        model_images = sorted(glob(os.path.join(model_dir, "*")))
        mask_images = sorted(glob(os.path.join(mask_dir, "*")))
        idx = np.random.randint(len(model_images))

        model_img = load_image(model_images[idx])
        mask_img = load_image(mask_images[idx])

        mask = mask_img[:, :, :, 0]

        return (model_img, mask)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return np.random.random()


class MannequinToModelLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "lora_name": (
                    folder_paths.get_filename_list("loras"),
                    {"tooltip": "The name of the LoRA."},
                ),
                "pose_hint": (["front", "side", "back", "closeup"],),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("model_name", "pose_prompt", "pose_hint")
    FUNCTION = "get_model_params"
    CATEGORY = "FaishmeNodes"

    def get_model_params(self, lora_name, pose_hint):
        model_name = lora_name.split(".")[0]
        pose_prompt_dict = {
            "front": f"full body portrait of {model_name} in front view",
            "side": f"full body portrait of {model_name} in side view",
            "back": f"full body portrait of {model_name} in back view",
            "closeup": f"closeup portrait of {model_name}",
            "none": f"{model_name}",
        }
        return (model_name, pose_prompt_dict[pose_hint], pose_hint)


class FaishmeDebug:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "value": (any, {}),
                "commands": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("output",)
    FUNCTION = "debug"
    CATEGORY = "FaishmeNodes"
    OUTPUT_NODE = True

    def debug(self, value, commands):
        output = {"value": value}
        exec(commands)
        return (output["value"],)


class MoondreamNode:
    def __init__(self):
        self.device = torch.device("cuda")
        self.dtype = torch.float32

        # get_torch_device()
        self.moondream = None
        self.tokenizer = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "question": (
                    "STRING",
                    {"multiline": True, "default": "", "dynamicPrompts": False},
                ),
                "device": (["cpu", "cuda"],),
            },
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "run"

    CATEGORY = "FaishmeNodes"

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    def run(self, image, question, device):

        result = []

        if device[0] != self.device.type and device[0] == "cpu":
            self.device = torch.device(device[0])
            self.dtype = torch.float32
            self.moondream = None
        elif device[0] != self.device.type:
            device, dtype = detect_device()
            self.device = device
            self.dtype = dtype
            self.moondream = None

        if self.moondream == None:
            revision = "2024-08-26"
            self.tokenizer = AutoTokenizer.from_pretrained(
                "vikhyatk/moondream2", revision=revision
            )
            self.moondream = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2", trust_remote_code=True, revision=revision
            )
            self.moondream.eval()

        question = question[0]

        for i in range(len(image)):
            im = image[i]
            im = tensor2pil(im)

            image_embeds = self.moondream.encode_image(im)

            # streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

            res = self.moondream.answer_question(image_embeds, question, self.tokenizer)

            result.append(res)

        return (result,)


NODE_CLASS_MAPPINGS = {
    "Load Fashion Model": LoadFashionModel,
    "Faishme Debug": FaishmeDebug,
    "Faishme Moondream": MoondreamNode,
    "Faishme Mannequin to Model Loader": MannequinToModelLoader,
}
