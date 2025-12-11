import uuid
import os
import folder_paths
from diffsynth.pipelines.qwen_image import (
    QwenImagePipeline, ModelConfig,
    QwenImageUnit_Image2LoRAEncode, QwenImageUnit_Image2LoRADecode
)
from modelscope import snapshot_download
from safetensors.torch import save_file
import torch
from PIL import Image
import numpy as np

class RunningHub_ImageQwenI2L_Loader_Style:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

            }
        }

    RETURN_TYPES = ('RH_QwenImageI2LPipeline', )
    RETURN_NAMES = ('QwenImageI2LPipeline', )
    FUNCTION = "load"
    CATEGORY = "RunningHub/ImageQwenI2L"

    def __init__(self):
        self.vram_config_disk_offload = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": "disk",
            "onload_device": "disk",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
        # self.encoder_path = os.path.join(folder_paths.models_dir, 'DiffSynth-Studio', 'General-Image-Encoders')
        # self.i2l_path = os.path.join(folder_paths.models_dir, 'DiffSynth-Studio', 'Qwen-Image-i2L')
        # self.processor_path = os.path.join(folder_paths.models_dir, 'DiffSynth-Studio', 'Qwen-Image-Edit')

    def load(self):
        pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(model_id='DiffSynth-Studio/General-Image-Encoders', origin_file_pattern="SigLIP2-G384/model.safetensors", **self.vram_config_disk_offload),
                ModelConfig(model_id='DiffSynth-Studio/General-Image-Encoders', origin_file_pattern="DINOv3-7B/model.safetensors", **self.vram_config_disk_offload),
                ModelConfig(model_id='DiffSynth-Studio/Qwen-Image-i2L', origin_file_pattern="Qwen-Image-i2L-Style.safetensors", **self.vram_config_disk_offload),
            ],
            processor_config=ModelConfig(model_id='Qwen/Qwen-Image-Edit', origin_file_pattern="processor/"),
            vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
        )
        return (pipe, )

class RunningHub_ImageQwenI2L_LoraGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("RH_QwenImageI2LPipeline", ),
                "training_images": ("IMAGE", ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ('STRING', 'RH_Lora')
    RETURN_NAMES = ('lora_name', 'lora')
    FUNCTION = "generate"
    CATEGORY = "RunningHub/ImageQwenI2L"

    OUTPUT_NODE = True

    def tensor_2_pil(self, img_tensor):
        i = 255. * img_tensor.squeeze().cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img

    def __init__(self):
        self.lora_name = f"i2l_style_lora_{str(uuid.uuid4())}.safetensors"

    def generate(self, pipeline, training_images, **kwargs):
        training_images = [self.tensor_2_pil(image) for image in training_images]
        training_images = [image.convert("RGB") for image in training_images]
        lora_path = os.path.join(folder_paths.models_dir, 'loras', self.lora_name)
        with torch.no_grad():
            embs = QwenImageUnit_Image2LoRAEncode().process(pipeline, image2lora_images=training_images)
            lora = QwenImageUnit_Image2LoRADecode().process(pipeline, **embs)["lora"]
        save_file(lora, lora_path)
        return (self.lora_name, lora_path)

NODE_CLASS_MAPPINGS = {
    "RunningHub_ImageQwenI2L_Loader(Style)": RunningHub_ImageQwenI2L_Loader_Style,
    "RunningHub_ImageQwenI2L_LoraGenerator": RunningHub_ImageQwenI2L_LoraGenerator,
}

