import os
import gc
import math
import psutil
import gpustat
from glob import glob
import numpy as np
import pandas as pd
import torch
import folder_paths as comfy_paths
import node_helpers
from PIL import Image, ImageOps
from transformers import AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor

import folder_paths


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
    )


# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def load_image_check(image_path, width, height, default_size):
    if os.path.isdir(image_path) and os.path.ex:
        return
    i = Image.open(image_path)
    i = ImageOps.exif_transpose(i)
    image = i
    if not default_size:
        image = i.resize((width, height), Image.LANCZOS)
    image = image.convert("RGB")
    image_np = np.array(image).astype(np.float32) / 255.0
    del i
    del image
    image_tensor = torch.from_numpy(image_np)[None,]
    del image_np  # Remove NumPy array
    return (image_tensor, image_path)


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


Any = AnyType("*")


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
                "pose_hint": (
                    [
                        "full-front",
                        "full-side",
                        "full-back",
                        "upper-front",
                        "upper-side",
                        "upper-back",
                        "closeup",
                    ],
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("model_name", "pose_prompt", "pose_hint")
    FUNCTION = "get_model_params"
    CATEGORY = "FaishmeNodes"

    def get_model_params(self, lora_name, pose_hint):
        model_name = lora_name.split(".")[0]
        pose_prompt_dict = {
            "full-front": f"full body portrait of {model_name} in front view",
            "upper-front": f"upper body portrait of {model_name} in front view",
            "full-side": f"full body portrait of {model_name} in side view",
            "upper-side": f"upper body portrait of {model_name} in side view",
            "full-back": f"full body portrait of {model_name} in back view",
            "upper-back": f"upper body portrait of {model_name} in back view",
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
                "value": (Any, {}),
                "commands": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = (Any,)
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
        self.moondream = None
        self.device = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "question": (
                    "STRING",
                    {"multiline": True, "default": "", "dynamicPrompts": False},
                ),
                "device": (["cuda", "cpu"],),
            },
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "run"

    CATEGORY = "FaishmeNodes"

    def run(self, image, question, device):
        if device != self.device:
            self.device = device
            del self.moondream
            self.moondream = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                revision="2025-01-09",
                trust_remote_code=True,
                # Uncomment for GPU acceleration & pip install accelerate
                device_map={"": self.device},
            )
            self.moondream.eval()

        res = self.moondream.query(tensor2pil(image[0]), question)["answer"]
        return (res,)


class LoadImagesFromGlobList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pattern": ("STRING", {"default": ""}),
                "file_path": ("STRING", {"default": ""}),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 0,
                        "max": 8192,
                        "step": 1,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1536,
                        "min": 0,
                        "max": 8192,
                        "step": 1,
                    },
                ),
                "default_size": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "FILE PATH", "width", "height")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True, False, False)

    FUNCTION = "load_images"

    CATEGORY = "FaishmeNodes"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if "load_always" in kwargs and kwargs["load_always"]:
            return float("NaN")
        else:
            return hash(frozenset(kwargs))

    def load_images(
        self,
        pattern: str,
        file_path,
        width: int,
        height: int,
        default_size: bool,
    ):
        pattern, width, height, default_size = (
            pattern[0],
            width[0],
            height[0],
            default_size[0],
        )
        if file_path[0] != "":
            dir_files = file_path
        else:
            dir_files = glob(pattern)
            if len(dir_files) == 0:
                raise FileNotFoundError(f"No files in directory '{pattern}'.")

            # Filter files by extension
            valid_extensions = [".jpg", ".jpeg", ".png", ".webp"]
            dir_files = [
                f
                for f in dir_files
                if any(f.lower().endswith(ext) for ext in valid_extensions)
            ]

            dir_files = sorted(dir_files)

        images = []
        file_paths = []

        with ThreadPoolExecutor() as executor:
            images, file_paths = list(
                zip(
                    *executor.map(
                        load_image_check,
                        dir_files,
                        [width] * len(dir_files),
                        [height] * len(dir_files),
                        [default_size] * len(dir_files),
                    )
                )
            )

        gc.collect()

        return (images, file_paths, width, height)


class StackImages:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "stack_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 16, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "stack_size")

    FUNCTION = "stack_image"
    CATEGORY = "FaishmeNodes"
    OUTPUT_NODE = True

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)

    def stack_image(self, image, stack_size):
        return (
            [
                torch.cat(image[i : i + stack_size[0]], dim=0)
                for i in range(0, len(image), stack_size[0])
            ],
            stack_size,
        )


class UnstackImages:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "stack_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 16, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "unstack_image"
    CATEGORY = "FaishmeNodes"
    OUTPUT_NODE = True

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    def unstack_image(self, image, stack_size):
        result = []
        for img in image:
            result.extend(img.split(stack_size[0], dim=0))
        return (result,)


class StackLatents:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "stack_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 16, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "stack_size")

    FUNCTION = "stack_latent"
    CATEGORY = "FaishmeNodes"
    OUTPUT_NODE = True

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)

    def stack_latent(self, latent, stack_size):
        return (
            [
                {
                    key: torch.cat(
                        [x[key] for x in latent[i : i + stack_size[0]]], dim=0
                    )
                    for key, _ in latent[i].items()
                }
                for i in range(0, len(latent), stack_size[0])
            ],
            stack_size,
        )


class UnstackLatents:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "stack_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 16, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    FUNCTION = "unstack_latent"
    CATEGORY = "FaishmeNodes"
    OUTPUT_NODE = True

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    def unstack_latent(self, latent, stack_size):
        result = []
        for lat in latent:
            result.extend(latent.split(stack_size[0], dim=0))
        return (result,)


class RepeatImageRowsBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "repeat"

    CATEGORY = "FaishmeNodes"

    def repeat(self, image, amount):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        b, h, w, c = image.shape
        s = (
            image.reshape(b, 1, h, w, c)
            .repeat((1, amount, 1, 1, 1))
            .reshape(-1, h, w, c)
        )
        return (s,)


class RepeatLatentRowsBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "amount": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "repeat"

    CATEGORY = "FaishmeDebug"

    def repeat(self, samples, amount):
        s = samples.copy()
        s_in = samples["samples"]

        if len(s_in.shape) == 3:
            s_in = s_in.unsqueeze(0)
        b, c, h, w = s_in.shape
        s["samples"] = (
            s_in.reshape(b, 1, c, h, w)
            .repeat((1, amount, 1, 1, 1))
            .reshape(-1, c, h, w)
        )
        if "noise_mask" in samples and samples["noise_mask"].shape[0] > 1:
            masks = samples["noise_mask"]
            if masks.shape[0] < s_in.shape[0]:
                masks = masks.repeat(
                    math.ceil(s_in.shape[0] / masks.shape[0]), 1, 1, 1
                )[: s_in.shape[0]]
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(0)
            b, c, h, w = masks.shape
            s["noise_mask"] = (
                masks.reshape(b, 1, c, h, w)
                .repeat((1, amount, 1, 1, 1))
                .reshape(-1, c, h, w)
            )
        if "batch_index" in s:
            offset = max(s["batch_index"]) - min(s["batch_index"]) + 1
            s["batch_index"] = s["batch_index"] + [
                x + (i * offset) for i in range(1, amount) for x in s["batch_index"]
            ]
        return (s,)


MEMORY_DEBUG_IDX = 0


class MemoryDebug:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "value": (Any, {}),
            }
        }

    RETURN_TYPES = (Any, "STRING")
    RETURN_NAMES = ("value", "debug_info")
    FUNCTION = "debug_memory"
    CATEGORY = "FaishmeNodes"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (
        True,
        False,
    )

    def debug_memory(self, value):
        global MEMORY_DEBUG_IDX
        debug_info = f"Index: {MEMORY_DEBUG_IDX}\n"
        ram_used = psutil.virtual_memory().used
        gpu_stats = gpustat.GPUStatCollection.new_query()
        debug_info += f"System RAM used: {ram_used/1e9:.2f} GB\n"
        for idx, gpu in enumerate(gpu_stats):
            debug_info += f"GPU {idx} VRAM used: {gpu.memory_used/1024:.2f} GB\n"
        MEMORY_DEBUG_IDX += 1
        return (value, debug_info)


class FaishmeSaveImage:
    def __init__(self):
        self.delim = "_"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "file_paths": (
                    "STRING",
                    {"forceInput": True, "tooltip": "Save location of file"},
                ),
                "suffix": (
                    "STRING",
                    {"default": "gen", "tooltip": "Suffix identifier for the output"},
                ),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True
    INPUT_IS_LIST = True

    CATEGORY = "FaishmeNodes"
    DESCRIPTION = "Saves the input images to the provided directory."

    def save_images(self, images, file_paths, suffix="gen"):
        suffix = suffix[0]

        def _output_path(file_path, suffix, n_img):
            base, ext = os.path.splitext(file_path)

            # create directory
            os.makedirs(os.path.dirname(base), exist_ok=True)

            if suffix != "":
                suffix = "_" + suffix

            counter_start = None if n_img == 1 else 1
            existing_files = glob(base + suffix + "*")
            if len(existing_files) > 0:
                try:
                    counter_start = (
                        int(max(existing_files).split(self.delim)[-1].split(".")[0]) + 1
                    )
                except ValueError:
                    counter_start = 1
            if counter_start:
                return [
                    f"{base}{suffix}_{counter:04}{ext}"
                    for counter in range(counter_start, counter_start + n_img)
                ]
            else:
                return [base + suffix + ext] * n_img

        def _save_img(img, path):
            img.save(path)

        save_paths = [
            path
            for img, file_path in zip(images, file_paths)
            for path in _output_path(file_path, suffix, len(img))
        ]
        imgs = [tensor2pil(i) for img in images for i in img]
        with ThreadPoolExecutor() as executor:
            # dummy list comprehension to catch thread exceptions in main thread
            [x for x in executor.map(_save_img, imgs, save_paths)]

        return ()


class RepeatTensorBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "tensor": (Any, {}),
                "repeat": ("INT", {"default": 1, "min": 1, "max": 128}),
            }
        }

    RETURN_TYPES = (Any,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "repeat"
    CATEGORY = "FaishmeNodes"

    def repeat(self, tensor, repeat):
        assert isinstance(tensor, torch.Tensor)
        return (tensor.repeat(repeat, *([1] * (tensor.dim() - 1))),)


class RepeatBBOX:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "bbox": ("BBOX",),
                "repeat": ("INT", {"default": 1, "min": 1, "max": 128}),
            }
        }

    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bbox",)
    FUNCTION = "repeat"
    CATEGORY = "FaishmeNodes"

    def repeat(self, bbox, repeat):
        return (bbox * repeat,)

class FaishmeSplit:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "value": (Any, {}),
            }
        }

    RETURN_TYPES = (
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
    )
    RETURN_NAMES = (
        "output1",
        "output2",
        "output3",
        "output4",
        "output5",
        "output6",
        "output7",
        "output8",
        "output9",
    )
    FUNCTION = "split"
    CATEGORY = "FaishmeNodes"
    OUTPUT_NODE = True

    def split(self, value):
        return value


class FaishmeGemini:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "images": ("IMAGE",),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "FaishmeNodes"

    def run(self, images, system_prompt, user_prompt, api_key):
        from tenacity import (
            retry,
            stop_after_attempt,
            wait_exponential,
            wait_random,
            retry_if_exception,
        )
        from google.api_core.exceptions import InternalServerError, GoogleAPICallError
        from vertexai.generative_models import FinishReason
        from google import genai
        from google.genai import types
        from google.genai.types import HttpOptions
        from typing_extensions import TypedDict
        import json

        GOOGLE_API_KEY = api_key
        gemini_client = genai.Client(
            api_key=GOOGLE_API_KEY, http_options=HttpOptions(timeout=120 * 1000)
        )

        class ResponseSchema1(TypedDict):
            upperwear: str
            lowerwear: str

        @retry(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=1, max=20) + wait_random(0, 2),
            retry=(
                retry_if_exception(GoogleAPICallError)
                | retry_if_exception(InternalServerError)
            ),
        )
        def call_gemini1(content, model_name, system_prompt):
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=content,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                ),
            )
            return response

        assert len(images.shape) == 4, "Expected 4 dims"
        imgs = []
        for img in images:
            imgs.append(tensor2pil(img))
        content = [*imgs, user_prompt]

        response = call_gemini1(content, "gemini-2.5-flash", system_prompt)
        if (
            response.candidates[0].finish_reason == 3
            or response.candidates[0].finish_reason == FinishReason.RECITATION
        ):
            return None
        output = response.text
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Load Fashion Model": LoadFashionModel,
    "Faishme Debug": FaishmeDebug,
    "Faishme Moondream": MoondreamNode,
    "Faishme Mannequin to Model Loader": MannequinToModelLoader,
    "Faishme Load Image from Glob": LoadImagesFromGlobList,
    "Faishme Stack Images": StackImages,
    "Faishme Unstack Images": UnstackImages,
    "Faishme Stack Latents": StackLatents,
    "Faishme Unstack Latents": UnstackLatents,
    "Faishme Repeat Image Batch": RepeatImageRowsBatch,
    "Faishme Repeat Latent Batch": RepeatLatentRowsBatch,
    "Faishme Memory Debug": MemoryDebug,
    "Faishme Save Image": FaishmeSaveImage,
    "Faishme Repeat Tensor Batch": RepeatTensorBatch,
    "Faishme Repeat BBOX": RepeatBBOX,
    "Faishme Split": FaishmeSplit,
    "Faishme Gemini": FaishmeGemini,
}
