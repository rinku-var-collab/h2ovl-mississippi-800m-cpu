import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
import os
import torchvision.transforms as T
from PIL import Image
import sys


# Custom implementation for CPU
def custom_load_single_image(file_name, max_num=6, msac=False):
    def build_transform(input_size: int=448):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    image = Image.open(file_name).convert('RGB')
    transform = build_transform(448)
    pixel_values = transform(image).unsqueeze(0)  # Add batch dimension
    pixel_values = pixel_values.to(torch.float32)  # Ensure CPU tensor

    # No MSAC logic for CPU
    num_patches_list = [pixel_values.size(0)]
    return pixel_values, num_patches_list


def patch_huggingface_cached_module():
    """
    Dynamically patch `load_single_image` in the Hugging Face cached module.
    """
    # Locate the Hugging Face dynamically loaded module
    module_name = None
    for name in sys.modules:
        if "modelling_h2ovl_chat" in name:
            module_name = name
            break

    if not module_name:
        raise RuntimeError("Hugging Face module for `h2ovl-mississippi-800m` not loaded.")

    # Get the module reference
    core_module = sys.modules[module_name]

    # Replace the functions
    setattr(core_module, 'load_single_image', custom_load_single_image)
    # If needed, replace load_multi_images similarly


# Initialize the model
def initialize_model(model_path: str='h2oai/h2ovl-mississippi-800m'):
    #model_path = 'h2oai/h2ovl-mississippi-800m'
    os.environ["HF_HOME"] = "./tmp_huggingface_cache"

    # Load configuration
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.llm_config._attn_implementation = "eager"

    # Load model on CPU
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # Ensure float32 for CPU
        config=config,
        trust_remote_code=True
    ).eval()  # No `.cuda()`

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    # Generation configuration
    generation_config = {
        "max_new_tokens": 2048,
        "do_sample": True
    }

    return model, tokenizer, generation_config





# Patch the .cuda() calls to use .cpu()
def patch_cuda_to_cpu(model):
    """
    Dynamically patch all .cuda() calls in the chat method to use .cpu().
    """
    original_chat = model.chat

    def patched_chat(*args, **kwargs):
        """
        Intercept .cuda() calls inside the chat method.
        """
        # Backup and patch .cuda() method
        original_cuda = torch.Tensor.cuda

        def patched_cuda(self):
            return self.cpu()  # Replace .cuda() with .cpu()

        # Replace .cuda with the patched version
        torch.Tensor.cuda = patched_cuda

        try:
            # Call the original chat method
            return original_chat(*args, **kwargs)
        finally:
            # Restore the original .cuda() method
            torch.Tensor.cuda = original_cuda

    model.chat = patched_chat


# Chat with the model
def chat_with_model(model, tokenizer, generation_config, input_text, history=None):
    """
    Call the patched chat method of the model for text-based input.
    """
    response, history = model.chat(
        tokenizer=tokenizer,
        image_files=None,  # No images for text-only input
        question=input_text,
        generation_config=generation_config,
        history=history,
        return_history=True,
    )
    return response, history
