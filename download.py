# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import os
from diffusers import StableDiffusionImageVariationPipeline

def download_model():
    # do a dry run of loading the huggingface model, which will download weights at build time
    #Set auth token which is required to download stable diffusion model weights
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

    model = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers",
    revision="v2.0",
    )

if __name__ == "__main__":
    download_model()
