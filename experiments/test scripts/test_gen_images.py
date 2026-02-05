import argparse
import json
import pandas as pd
import gc
import os
from pathlib import Path
import torch
import traceback
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

def extract_code_and_robust(code_file,file_name:str):
    """
    Extracts the skill code from the file name
    """

    # Check if robust in string or not
    robust = "robust" in file_name
    skills = file_name[:-12] if robust else file_name[:-5]
    # Retrieve the skill code
    code = code_file.loc[code_file["skill"]==skills, "code"]
    # Check if the code is not empty
    if not code.empty: 
        code = code.iloc[0]
    else: 
        code = -1

    return code,robust

def extract_prompt_info(prompt:dict):
    """
    """

    # Get the id 
    prompt_number = prompt["id"].split("_")[1]
    # Get the level 
    prompt_level = prompt["level"]
    # Get the synthetic prompts
    synthetic_prompts = prompt["synthetic_prompts"]

    return prompt_number,prompt_level,synthetic_prompts

def set_output_name(model_name:str,skill_code,prompt_level:str,prompt_number:str, synthetic_prompt_number:int, output_folder_name:str, robust:bool):


    # Initialize the variable
    output_file_name = ""

    if robust:
        # Add the robust tag
        output_file_name = output_folder_name + "/" +  model_name + "_" + str(skill_code) + "_" + prompt_level + "_" + str(prompt_number) + "_" + str(synthetic_prompt_number) + "_robust"
    else: 
        output_file_name = output_folder_name + "/" +  model_name + "_" + str(skill_code) + "_" + prompt_level + "_" + str(prompt_number) + "_" + str(synthetic_prompt_number)

    return output_file_name


def generate_image_animagine(code_file,GPU="0"):
    """
    """
    # Set the json outputs directory (.json)
    json_dir = Path("../../outputs/prompts")
    # Get the list of json files
    json_files = [p for p in json_dir.glob("*") if p.is_file()]
    # Set the image output directory
    output_dir_name = "../../data/images_refactored/animagine"
    output_dir = Path(output_dir_name)
    # Get the list of files to generate
    output_files = [p._str for p in output_dir.glob("*") if p.is_file()]

    try: 
        # Import relevant packages
        from diffusers import StableDiffusionXLPipeline
        # Setup the pipe
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "cagliostrolab/animagine-xl-4.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            custom_pipeline="lpw_stable_diffusion_xl",
            add_watermarker=False
        ).to(f"cuda:{GPU}")
        # Set the elements 
        model_name = "animagine"
        # Display a message
        print("\nGenerating images with animagine.\n")
        # Loop through the files
        for i,json_file in enumerate(json_files):
            print(json_file)
            # Extract the skill code
            skill_code, robust = extract_code_and_robust(code_file,os.path.basename(json_file))
            # Initialize the collection
            elements = dict()
            # Open the file
            with open(json_file, "rb") as file:
                # Load the elements into a dict
                elements = json.load(file)
                # Loop through the prompts
                for prompt in elements["prompts"]:
                    # Get the prompt infos
                    prompt_number, prompt_level, synthetic_prompts = extract_prompt_info(prompt)
                    # Loop through the synthetic prompts
                    for j,gen_prompt in enumerate(synthetic_prompts):
                        # Set the output file name
                        output_file_name = set_output_name(model_name, skill_code, 
                                                           prompt_level,prompt_number,j,output_dir_name, robust)
                        # Check if the file doesn't exist
                        if output_file_name+".png" not in output_files:
                            # Display the output file name
                            print(f"\nOutput file name: {output_file_name}")
                            # Run inference ; generate the image
                            image = pipe(
                                gen_prompt, 
                                width=1024,
                                height=1024, 
                                guidance_scale=5, 
                                num_inference_steps=28, 
                                generator=torch.Generator(f"cuda:{GPU}").manual_seed(42) #Setting the seed to be more deterministic
                            ).images[0]
                            # Save image
                            image.save(output_file_name+".png")
                            # Display success message
                            print(f"\nImage generated for : {output_file_name}. Prompt: {gen_prompt}")
        # Free memory
        del image, pipe
        # Collect garbage
        gc.collect()
        # Empty cuda cache
        torch.cuda.empty_cache()
        # Collect garbage
        torch.cuda.ipc_collect()
        
        return 1
    except Exception:
        # Display exception
        traceback.print_exc()

def generate_image_stable_diffusion(code_file,GPU="0"):
    """
    """
    # Set the json outputs directory (.json)
    json_dir = Path("../../outputs/prompts")
    # Get the list of json files
    json_files = [p for p in json_dir.glob("*") if p.is_file()]
    json_files.sort()
    # Set the image output directory
    output_dir_name = "../../data/images_refactored/stable_diffusion"
    output_dir = Path(output_dir_name)
    # Get the list of files to generate
    output_files = [p._str for p in output_dir.glob("*") if p.is_file()]
    output_files.sort()

    try: 
        # Import relevant packages
        from diffusers import StableDiffusionXLPipeline
        from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl
        # Setup the pipe
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                                 torch_dtype=torch.float16, 
                                                 use_safetensors=True,
                                                 variant="fp16").to(f"cuda:{GPU}")
        
        
        # Set the elements 
        model_name = "stable_diffusion"
        # Display a message
        print("\nGenerating images with stable diffusion.\n")
        # Loop through the files
        for i,json_file in enumerate(json_files):
            print(json_file)
            # Extract the skill code
            skill_code, robust = extract_code_and_robust(code_file,os.path.basename(json_file))
            # Initialize the collection
            elements = dict()
            # Open the file
            with open(json_file, "rb") as file:
                # Load the elements into a dict
                elements = json.load(file)
                # Loop through the prompts
                for prompt in elements["prompts"]:
                    # Get the prompt infos
                    prompt_number, prompt_level, synthetic_prompts = extract_prompt_info(prompt)
                    # Loop through the synthetic prompts
                    for j,gen_prompt in enumerate(synthetic_prompts):
                        # Set the output file name
                        output_file_name = set_output_name(model_name=model_name, 
                                                           skill_code=skill_code, 
                                                           prompt_level=prompt_level,
                                                           prompt_number=prompt_number,
                                                           synthetic_prompt_number=j,
                                                           output_folder_name=output_dir_name, 
                                                           robust=robust)
                        # Check if the file doesn't exist
                        if output_file_name+".png" not in output_files:
                            # Display the output file name
                            print(f"\nOutput file name: {output_file_name}")
                            # Run inference ; generate the image
                            with torch.no_grad():
                                # Adding support for long prompts
                                (prompt_embeds, 
                                 prompt_neg_embeds,
                                 pooled_prompt_embeds,
                                 negative_pooled_prompt_embeds) = get_weighted_text_embeddings_sdxl(
                                     pipe,
                                     prompt = gen_prompt,
                                )
                                image = pipe(
                                    prompt_embeds=prompt_embeds,
                                    negative_prompt_embeds=prompt_neg_embeds,
                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                    width=1024,
                                    height=1024, 
                                    guidance_scale=5, 
                                    num_inference_steps=50, 
                                    generator=torch.Generator(f"cuda:{GPU}").manual_seed(42) #Setting the seed to be more deterministic
                                ).images[0]
                                # Save image
                                image.save(output_file_name+".png")
                            # Display success message
                            print(f"\nImage generated for : {output_file_name}. Prompt: {gen_prompt}")
        # Free memory
        del image, pipe
        # Collect garbage
        gc.collect()
        # Empty cuda cache
        torch.cuda.empty_cache()
        # Collect garbage
        torch.cuda.ipc_collect()
        
        return 1
    except Exception:
        # Display exception
        traceback.print_exc()


def generate_z_image(code_file,GPU="0"):
    """
    """
    # Set the json outputs directory (.json)
    json_dir = Path("../../outputs/prompts")
    # Get the list of json files
    json_files = [p for p in json_dir.glob("*") if p.is_file()]
    # Set the image output directory
    output_dir_name = "../../data/images_refactored/z_image_turbo"
    output_dir = Path(output_dir_name)
    # Get the list of files to generate
    output_files = [p._str for p in output_dir.glob("*") if p.is_file()]

    try: 
        # Import relevant packages
        from diffusers import ZImagePipeline
        from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl
        # Setup the pipe
        pipe = ZImagePipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                                 torch_dtype=torch.float16, 
                                                 use_safetensors=True,
                                                 variant="fp16").to(f"cuda:{GPU}")
        
        
        # Set the elements 
        model_name = "z_image_turbo"
        # Display a message
        print("\nGenerating images with z_image_turbo.\n")
        # Loop through the files
        for i,json_file in enumerate(json_files):
            print(json_file)
            # Extract the skill code
            skill_code, robust = extract_code_and_robust(code_file,os.path.basename(json_file))
            # Initialize the collection
            elements = dict()
            # Open the file
            with open(json_file, "rb") as file:
                # Load the elements into a dict
                elements = json.load(file)
                # Loop through the prompts
                for prompt in elements["prompts"]:
                    # Get the prompt infos
                    prompt_number, prompt_level, synthetic_prompts = extract_prompt_info(prompt)
                    # Loop through the synthetic prompts
                    for j,gen_prompt in enumerate(synthetic_prompts):
                        # Set the output file name
                        output_file_name = set_output_name(model_name, skill_code, 
                                                           prompt_level,prompt_number,j, output_dir_name, robust)
                        # Check if the file doesn't exist
                        if output_file_name+".png" not in output_files:
                            # Display the output file name
                            print(f"\nOutput file name: {output_file_name}")
                            # Run inference ; generate the image
                            with torch.no_grad():
                                # Adding support for long prompts
                                (prompt_embeds, 
                                 prompt_neg_embeds,
                                 pooled_prompt_embeds,
                                 negative_pooled_prompt_embeds) = get_weighted_text_embeddings_sdxl(
                                     pipe,
                                     prompt = gen_prompt,
                                )
                                image = pipe(
                                    prompt_embeds=prompt_embeds,
                                    negative_prompt_embeds=prompt_neg_embeds,
                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                    width=1024,
                                    height=1024, 
                                    guidance_scale=5, 
                                    num_inference_steps=50, 
                                    generator=torch.Generator(f"cuda:{GPU}").manual_seed(42) #Setting the seed to be more deterministic
                                ).images[0]
                                # Save image
                                image.save(output_file_name+".png")
                            # Display success message
                            print(f"\nImage generated for : {output_file_name}. Prompt: {gen_prompt}")
        # Free memory
        del image, pipe
        # Collect garbage
        gc.collect()
        # Empty cuda cache
        torch.cuda.empty_cache()
        # Collect garbage
        torch.cuda.ipc_collect()
        
        return 1
    except Exception:
        # Display exception
        traceback.print_exc()


def initialize_parser():
    """
    Initializes the argument parser. 
    """

    # Initializing the parser 
    parser = argparse.ArgumentParser(description="Test script for generating images.")

    # Add arguments 
    parser.add_argument("--gpu", choices=["0", "1", "2"], default="0", help="Preferred GPU number.")
    parser.add_argument("--model", choices=["all", 
                                            "firelfy", 
                                            "dalle", 
                                            "kandinsky", 
                                            "runway", 
                                            "stable_cascade", 
                                            "stable_diffusion", 
                                            "z_image"], default="stable_diffusion"),
    

    return parser


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
    # Set the model 
    MODEL = parser.parse_args().model
    # Load the code file
    code_file = pd.read_csv("../../skills_code.csv", encoding="utf-8")

    if MODEL=="animagine":
        # Generate images with animagine
        generate_image_animagine(code_file,GPU)
    elif MODEL=="stable_diffusion":
        # Generate images with stable diffusion
        generate_image_stable_diffusion(code_file,GPU)




if __name__=="__main__":
    # Load the main
    main()