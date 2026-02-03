import argparse
import gc
import os
import time
import torch
import traceback
from accelerate import Accelerator
from diffusers import  DiffusionPipeline, StableCascadeCombinedPipeline, \
    ZImagePipeline, StableDiffusionXLPipeline
from dotenv import load_dotenv


def load_environment() -> str:
    """
    Loads the environment from the .env file in os system variables.
    
    return: (str) Access token. 
    
    """

    # Load the environment files
    load_dotenv("../../.env")
    # Load the access token
    access_token = os.environ["HF_TOKEN"] if os.environ["HF_TOKEN"] else False

    return access_token


def test_stable_cascade(prompt:str, access_token:str, negative_prompt="", GPU="0"):
    """
    Tests the Stable Cascade model. 
    Steps : 
    1. Initialize the pipe
    2. Runs inference (generate an image from the prompt). 
    3. Returns the status.
    
    (str) prompt: The prompt for image generation.
    (str) access_token: Access token for HuggingFace.
    (str) negative_prompt: The negative prompt.
    """

    # Initialize the start time
    start_time = time.time()
    
    try:
        # Initialize the pipe
        pipe = StableCascadeCombinedPipeline.from_pretrained("stabilityai/stable-cascade", 
                                                             variant="bf16", 
                                                             torch_dtype=torch.bfloat16, 
                                                             access_token=access_token).to(f"cuda:{GPU}")
        # Run inference
        pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=10,
            prior_num_inference_steps=20,
            prior_guidance_scale=3.0,
            width=1024,
            height=1024,
        ).images[0]

        # Display the elapsed time
        print(f"Elapsed time for stable cascade [bf16]: {round(time.time()-start_time,2)} seconds.")
        # Free memory
        del(pipe)
        # Collect garbage
        gc.collect()
        # Empty cuda cache
        torch.cuda.empty_cache()
        # Collect cuda garbage
        torch.cuda.ipc_collect()

        return 1

    except Exception:
        # Display trace
        traceback.print_exc()
        return -1

def test_qwen_image(prompt:str, access_token:str, negative_prompt=""):
    """
    Tests the QWEN-Image model. 
    Steps : 
    1. Initialize the pipe
    2. Runs inference (generate an image from the prompt). 
    3. Returns the status.
    
    (str) prompt: The prompt for image generation.
    (str) access_token: Access token for HuggingFace.
    (str) negative_prompt: The negative prompt.
    """

    # Initialize the accelerator
    accelerator = Accelerator()

    # Initialize the start time 
    start_time = time.time() 


    try:
        # Initialize the pipe
        pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", 
                                                 torch_dtype=torch.bfloat16, 
                                                 access_token=access_token)
        
        # Pipeline ---> Distributed GPUs
        pipe = accelerator.prepare(pipe)

        # Find where the Unet lives
        noise_device = pipe.transformer.device

        # Creating the generator on that device 
        generator = torch.Generator(device=noise_device).manual_seed(42)
        
        # Set image size
        width = height = 1024
        # Generate the image 
        image = pipe(
            prompt, 
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=10, 
            true_cfg_scale=3.0, 
            generator=generator
        ).images[0]

        if accelerator.is_main_process:
            # Display elapsed time
            print(f"Elapsed time for QWEN [bfloat16] : {round(time.time()-start_time,2)} seconds.")
        # Free memory
        del(image)
        del(pipe)
        # Collect garbage
        gc.collect()
        # Empty cuda cache
        torch.cuda.empty_cache()
        # Collect cuda garbage
        torch.cuda.ipc_collect()

        return 1

    except Exception: 
        # Display trace 
        traceback.print_exc()
        return -1


def test_z_image(prompt:str, access_token:str, negative_prompt="", GPU="0"):
    """
    Tests the Z-Image model. 
    Steps : 
    1. Initialize the pipe
    2. Runs inference (generate an image from the prompt). 
    3. Returns the status.
    
    (str) prompt: The prompt for image generation.
    (str) access_token: Access token for HuggingFace.
    (str) negative_prompt: The negative prompt.
    """

    # Ininitialize the start time 
    start_time = time.time()

    try: 
        # Setup the pipe
        pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", 
                                              torch_dtype=torch.bfloat16, 
                                              access_token=access_token,
                                              low_cpu_mem_usage=False).to(f"cuda:{GPU}")
        # Enable cpu offload (reducing memory)
        pipe.enable_model_cpu_offload()

        # Run inference
        image = pipe(
            prompt=prompt, 
            height=1024,
            width=1024, 
            num_inference_steps=9, 
            guidance_scale=0.0, 
            generator=torch.Generator("cuda:0").manual_seed(42)
        ).images[0]

        # Display elapsed time
        print(f"Elapsed time : {round(time.time()-start_time,2)} seconds.")
        # Free memory
        del(image)
        del(pipe)
        # Collect garbage 
        gc.collect()
        # Empty cuda cache
        torch.cuda.empty_cache()
        # Collect cuda garbage
        torch.cuda.ipc_collect()

        return 1
    
    except Exception:
        # Display error message
        traceback.print_exc()

        return -1


def test_stable_diffusion(prompt:str, access_token:str, negative_prompt="", GPU="0"):
    """
    Tests the Stable Diffusion model. 
    Steps : 
    1. Initialize the pipe
    2. Runs inference (generate an image from the prompt). 
    3. Returns the status.
    
    (str) prompt: The prompt for image generation.
    (str) access_token: Access token for HuggingFace.
    (str) negative_prompt: The negative prompt.
    """

    # Initialize the start time
    start_time = time.time()

    try:
        # Setup the pipe
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                                 torch_dtype=torch.float16, 
                                                 use_safetensors=True, 
                                                 access_token=access_token,
                                                 variant="fp16", ).to(f"cuda:{GPU}")
        # Run inference 
        images = pipe(
            prompt=prompt,
            width=1024, 
            height=1024, 
            num_inference_steps=10).images[0]
        # Display elapsed time 
        print(f"Elapsed time : {round(time.time()-start_time,2)} seconds.")
        # Free memory
        del(images)
        del(pipe)
        # Collect garbage
        gc.collect()
        # Empty cuda cache
        torch.cuda.empty_cache()
        # Collect garbage
        torch.cuda.ipc_collect()

        return 1
    
    except Exception:
        # Display error message 
        traceback.print_exc()

        return -1


def test_kandinsky(prompt:str, access_token:str, negative_prompt="", GPU="0"):
    """
    Tests the Kandinsky model. 
    Steps : 
    1. Initialize the pipe
    2. Runs inference (generate an image from the prompt). 
    3. Returns the status.
    
    (str) prompt: The prompt for image generation.
    (str) access_token: Access token for HuggingFace.
    (str) negative_prompt: The negative prompt.
    """

    # Initialize the start time 
    start_time = time.time()

    try:

        # Load the prior 
        prior = DiffusionPipeline.from_pretrained( "kandinsky-community/kandinsky-2-2-prior", 
                                                  torch_dtype=torch.bfloat16).to(f"cuda:{GPU}")
        # Load the decoder
        decoder = DiffusionPipeline.from_pretrained( "kandinsky-community/kandinsky-2-2-decoder", 
                                                    torch_dtype=torch.bfloat16).to(f"cuda:{GPU}")
        # Get the prior output
        prior_output = prior(prompt=prompt, negative_prompt=negative_prompt)
        # Get the image from embeddings
        image = decoder(image_embeds=prior_output.image_embeds, 
                        negative_image_embeds=prior_output.negative_image_embeds)
        
        print(f"Elapsed time : {round(time.time()-start_time,2)} seconds.")
        # Free memory
        del image, prior_output, decoder, prior
        #del(pipe)
        # Collect garbage
        gc.collect()
        # Empty cuda cache
        torch.cuda.empty_cache()
        # Collect cuda garbage
        torch.cuda.ipc_collect()

        return 1
    
    except Exception:
        # Display error message 
        traceback.print_exc()

        return -1
    

def test_animagine_XL(prompt:str, access_token:str, negative_prompt="", GPU="0"):
    """
    Tests the Animagine model. 
    Steps : 
    1. Initialize the pipe
    2. Runs inference (generate an image from the prompt). 
    3. Returns the status.
    
    (str) prompt: The prompt for image generation.
    (str) access_token: Access token for HuggingFace.
    (str) negative_prompt: The negative prompt.
    """

    # Initialize the start time 
    start_time = time.time()

    try: 
        # Setup the pipe
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "cagliostrolab/animagine-xl-4.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            custom_pipeline="lpw_stable_diffusion_xl",
            add_watermarker=False
        ).to(f"cuda:{GPU}")
        # Run inference
        image = pipe(
            prompt, 
            negative_prompt=negative_prompt, 
            width=1024,
            height=1024, 
            guidance_scale=5, 
            num_inference_steps=10
        )

        # Display elapsed time
        print(f"Elapsed time: {round(time.time()-start_time,2)} seconds.")
        # Free memory
        del(image)
        del(pipe)
        # Collect garbage
        gc.collect()
        # Empty cuda cache
        torch.cuda.empty_cache()
        # Collect garbage
        torch.cuda.ipc_collect()

        return 1
    except Exception: 
        # Display error message
        traceback.print_exc()
        return -1
    

def initialize_parser():
    """
    """

    # Initializing the parser 
    parser = argparse.ArgumentParser(description="Test script for evaluating gen models.\nLoads the models and runs \
                                     a 10-step inference, generating a 1024x1024 image, without a negative prompt.\n\
                                     The same prompt is used for every model.")

    # Add arguments 
    parser.add_argument("--gpu", choices=["0", "1", "2"], default="0", help="Preferred GPU number.")

    return parser

def test_models(access_token,models=[""], prompt="", GPU="0"):
    """
    Loops through the list of models and test each independently. 
    """

    # Test prompt
    prompt = "an image of a shiba inu, donning a spacesuit and helmet"
    negative_prompt = ""

    # Loop through the models
    for model in models: 
        # Stable cascade 
        if model=="stabilityai/stable-cascade": 
            # Test the stable cascade model
            if test_stable_cascade(prompt, access_token,negative_prompt,GPU)==-1:
                # Failure
                print("\n\nStable cascade : KO\n\n")
            else:
                # Success
                print("\n\nStable cascade : OK\n\n")
        # Stable diffusion
        elif model=="stabilityai/stable-diffusion-xl-base-1.0":
            # Test the stable diffusion model
            if test_stable_diffusion(prompt,access_token,negative_prompt,GPU)==-1:
                # Failure 
                print("\n\nStable Diffusion: KO\n\n")
            else:
                # Success
                print("\n\nStable Diffusion: OK\n\n")
        # Z-turbo
        elif model=="Tongyi-MAI/Z-Image-Turbo": 
            # Test the Z-image Turbo model
            if test_z_image(prompt,access_token,negative_prompt,GPU)==-1:
                # Failure 
                print("\n\nZ-image Turbo: KO\n\n")
            else: 
                # Success
                print("\n\nZ-Image Turbo : OK\n\n")
        # Animagine XL
        elif model=="cagliostrolab/animagine-xl-4.0":
            # Test the Animagine model
            if test_animagine_XL(prompt, access_token, negative_prompt,GPU)==-1:
                # Failure 
                print("\n\nAnimagine: KO\n\n")
            else: 
                # Success
                print("\n\nAnimagine: OK\n\n")
        # Kandinsky
        elif model=="kandinsky-community/kandinsky-2-2-prior": 
            pass
            # Test the Kandinsky model
            if test_kandinsky(prompt,access_token,negative_prompt,GPU)==-1:
                # Failure
                print("\n\nKandinsky: KO\n\n")
            else:
                # Success
                print("\n\nKandinsky: OK\n\n")
        # QWEN-Image
        """elif model=="Qwen/Qwen-Image":
            # Test the QWEN-Image model
            if test_qwen_image(prompt,access_token,negative_prompt)==-1:
                # Failure
                print("\n\nQWEN-Image: KO\n\n")
            else:
                # Success
                print("\n\nQWEN-Image: OK\n\n")"""


    return 1

def main():
    """
    Main script
    """
    # Free memory
    gc.collect()
    # Empty cuda cache
    torch.cuda.empty_cache()
    # Collect garbage
    torch.cuda.ipc_collect()

    # Parse arguments 
    parser = initialize_parser()
    # Set the GPU
    GPU = parser.parse_args().gpu

    # Load environment and get access token
    access_token = load_environment()
    # Test prompt
    PROMPT = "an image of a shiba inu, donning a spacesuit and helmet"
    # Set the list of models
    MODELS = [
        "Tongyi-MAI/Z-Image-Turbo",
        "stabilityai/stable-cascade", 
        "stabilityai/stable-diffusion-xl-base-1.0",
        "Qwen/Qwen-Image", 
        "kandinsky-community/kandinsky-2-2-prior",
        "cagliostrolab/animagine-xl-4.0"
        ]
    # Load the models 
    test_models(access_token, MODELS, PROMPT, GPU)


if __name__=="__main__":
    # Load the main
    main()