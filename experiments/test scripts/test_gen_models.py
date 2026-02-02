import os
import time
import torch
import traceback
from diffusers import StableCascadeCombinedPipeline, DiffusionPipeline
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
                                                             access_token=access_token).to("cuda")
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

        return 1


    except Exception:
        # Display trace
        traceback.print_exc()
        return -1

def test_qwen_image(prompt:str, access_toke:str, negative_prompt=""):
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
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"
    
    # Initialize the start time 
    start_time = time.time() 
    

    try:
        # Initialize the pipe
        pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", 
                                                 torch_dtype=torch_dtype).to(device)
        
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
            generator=torch.Generator(device="cuda").manual_seed(42)
        ).images[0]
        # Display elapsed time
        print(f"Elapsed time for QWEN [bfloat16] : {round(time.time()-start_time,2)} seconds.")
        # Free memory
        del(pipe)

        return 1

    except Exception: 
        # Display trace 
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
            if test_stable_cascade(prompt, access_token, negative_prompt)==-1:
                # Success
                print("Stable cascade : KO")
            else:
                # Failure 
                print("Stable cascade : OK")
        # QWEN-Image
        elif model=="Qwen/Qwen-Image":
            # Test the QWEN-Image model
            if test_qwen_image(prompt,access_token, negative_prompt)==-1:
                # Success
                print("QWEN-Image: KO")
            else:
                # Failure 
                print("QWEN-Image: OK")
    
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
    MODELS = ["stabilityai/stable-cascade", 
              "Qwen/Qwen-Image"]
    # Load the models 
    test_models(access_token, MODELS, PROMPT)
        

if __name__=="__main__":
    # Load the main
    main()