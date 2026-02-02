import gc
import os
import time
import torch
import traceback
from diffusers import  DiffusionPipeline, StableCascadeCombinedPipeline, ZImagePipeline, AutoPipelineForText2Image
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


def test_stable_cascade(prompt:str, access_token:str, negative_prompt=""):
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
                                                             access_token=access_token).to("cuda:0")
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

    # Load the pipeline
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda:0"
    else:
        torch_dtype = torch.float32
        device = "cpu"
    
    # Initialize the start time 
    start_time = time.time() 


    try:
        # Initialize the pipe
        pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", 
                                                 torch_dtype=torch_dtype, 
                                                 access_token=access_token).to(device)
        
        # Set image size
        width, height = 1024
        # Generate the image 
        image = pipe(
            prompt, 
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=10, 
            true_cfg_scale=3.0, 
            generator=torch.Generator(device="cuda:0").manual_seed(42)
        ).images[0]
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


def test_z_image(prompt:str, access_token:str, negative_prompt=""):
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
                                              low_cpu_mem_usage=False).to("cuda:0")
        # Run inference
        image = pipe(
            prompt=prompt, 
            height=1024,
            width=1024, 
            num_inference_steps=10, 
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


def test_stable_diffusion(prompt:str, access_token:str, negative_prompt=""):
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
                                                 variant="fp16", ).to("cuda:0")
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


def test_kandinsky(prompt:str, access_token:str, negative_prompt=""):
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
        # Setup the pipeline
        pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", 
                                                         torch_dtype=torch.float16).to("cuda:0")
        # Run inference
        image = pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            prior_guidance_scale =1.0, height=1024, width=1024).images[0]
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
    

#def test_animagine_XL(prompt:str, access_token:str, negative_prompt=""):
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
        ).to("cuda:0")
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

        return 1
    except Exception: 
        # Display error message
        traceback.print_exc()
        return -1



def test_models(access_token,models=[""], prompt=""):
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
            if test_stable_cascade(prompt, access_token,negative_prompt)==-1:
                # Failure
                print("\n\nStable cascade : KO\n\n")
            else:
                # Success
                print("\n\nStable cascade : OK\n\n")
        # Stable diffusion
        elif model=="stabilityai/stable-diffusion-xl-base-1.0":
            # Test the stable diffusion model
            if test_stable_diffusion(prompt,access_token,negative_prompt)==-1:
                # Failure 
                print("\n\nStable Diffusion: KO\n\n")
            else:
                # Success
                print("\n\nStable Diffusion: OK\n\n")
        # Z-turbo
        elif model=="Tongyi-MAI/Z-Image-Turbo": 
            # Test the Z-image Turbo model
            if test_z_image(prompt,access_token,negative_prompt)==-1:
                # Failure 
                print("\n\nZ-image Turbo: KO\n\n")
            else: 
                # Success
                print("\n\nZ-Image Turbo : OK\n\n")
        # Kandinsky
        elif model=="kandinsky-community/kandinsky-2-2-decoder": 
            pass
            # Test the Kandinsky model
            if test_kandinsky(prompt,access_token,negative_prompt)==-1:
                # Failure
                print("Kandinsky: KO")
            else:
                # Success
                print("Kandinsky: OK")
        # QWEN-Image
        """elif model=="Qwen/Qwen-Image":
            # Test the QWEN-Image model
            if test_qwen_image(prompt,access_token,negative_prompt)==-1:
                # Failure
                print("\n\nQWEN-Image: KO")
            else:
                # Success
                print("\n\nQWEN-Image: OK")"""

        # Animagine XL
        """elif model=="cagliostrolab/animagine-xl-4.0":
            # Test the Animagine model
            if test_animagine_XL(prompt, access_token, negative_prompt)==-1:
                # Failure 
                print("Animagine: KO")
            else: 
                # Success
                print("Animagine: OK")"""


    return 1

def main():
    """
    Main script
    """
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
        "kandinsky-community/kandinsky-2-2-decoder",
        "cagliostrolab/animagine-xl-4.0"
        ]
    # Load the models 
    test_models(access_token, MODELS, PROMPT)
        

if __name__=="__main__":
    # Load the main
    main()