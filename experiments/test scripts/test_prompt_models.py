import argparse
import gc
import os
import traceback
import time
import torch
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,\
                        AutoConfig
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


def test_qwen(prompt:str, access_token:str, negative_prompt:str, GPU="0"):
    """
    Tests the Qwen model. 
    Steps : 
    1. Loads the model and tokenizer
    2. Prepares the inputs. 
    3. Performs inference (with max new tokens = 4000).
    4. Retrieves the output and displays previous and currrent prompt.
    
    (str) prompt: The prompt for image generation.
    (str) access_token: Access token for HuggingFace.
    (str) negative_prompt: The negative prompt.
    """

    # Initialize the time
    start_time = time.time()
    # Set the model name 
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    try: 
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                  access_token=access_token)
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                     torch_dtype="auto", 
                                                     device_map="auto")
        # Prepare the model input
        messages = [
            {"role": "user", 
             "content": prompt}
        ]
        # Tokenize the text
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True,
        )
        # Set the model inputs 
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        # Conduct text completion
        generated_ids = model.generate(
            **model_inputs, 
            max_new_tokens=4000
        )
        # Retrieves output
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        content = tokenizer.decode(output_ids, skip_special_tokens=True)

        # Display both prompts
        print(f"Original prompt: {prompt}.\nGenerated prompt: {content}")

        # Display elpased time
        print(f"Elapsed time: {round(time.time()-start_time,2)} seconds.")

        # Free memory
        del content, output_ids, generated_ids, model_inputs, text, messages, model, tokenizer
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
        # Return failure 
        return -1

def test_mindlink(prompt:str, access_token:str, negative_prompt:str, GPU="0"):
    """
    Tests the Mindlink model. 
    Steps : 
    1. Loads the model and tokenizer
    2. Prepares the inputs. 
    3. Performs inference (with max new tokens = 4000).
    4. Retrieves the output and displays previous and currrent prompt.
    
    (str) prompt: The prompt for image generation.
    (str) access_token: Access token for HuggingFace.
    (str) negative_prompt: The negative prompt.
    """

    # Initialize the time
    start_time = time.time()
    # Set the model name 
    model_name = "Skywork/MindLink-32B-0801"
    # Quantization config
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True, 
        llm_int8_enable_fp32_cpu_offload=True
    )

    try: 
        # Load the model config
        config = AutoConfig.from_pretrained("Skywork/MindLink-32B-0801")
        print("\nLoading the tokenizer....\n")
        # Loading the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("\nLoading the model....\n")
        # Loading the model
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            quantization_config=quant_config, 
            device_map=f"cuda:{GPU}"
        )
        # Initialize the input
        messages = [
            {"role": "user", 
             "content": prompt}
            ]
        # Tokenize the text
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True,
        )
        # Set the model inputs : moving only the needed tensors to GPU
        model_inputs = tokenizer([text], return_tensors="pt")
        model_inputs = {k:v.to(model.device) for k,v in model_inputs.items()}
        # Generate the answer 
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs, 
                max_new_tokens=400, 
                use_cache=False
            )
        generated_ids = [
            output_ids[len(input_ids):]\
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Get the response
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Display both prompts
        print(f"Original prompt: {prompt}.\nGenerated prompt: {response}")

        # Display elpased time
        print(f"Elapsed time: {round(time.time()-start_time,2)} seconds.")

        # Free memory
        del response, generated_ids, model_inputs, text, messages, model, tokenizer
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
        # Return failure 
        return -1

def test_deepseek(prompt:str, access_token:str, negative_prompt:str, GPU="0"):
    """
    Tests the Deepseek model. 
    Steps : 
    1. Loads the model and tokenizer
    2. Prepares the inputs. 
    3. Performs inference (with max new tokens = 4000).
    4. Retrieves the output and displays previous and currrent prompt.
    
    (str) prompt: The prompt for image generation.
    (str) access_token: Access token for HuggingFace.
    (str) negative_prompt: The negative prompt.
    """

    # Initialize model name 
    model_name  = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    # Initialize the start time
    start_time = time.time()

    try:
        print("\nLoading the tokenizer....\n")
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load model 
        print("\nLoading the model....\n")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # Setup the messages
        messages = [
            {"role": "user", 
             "content": prompt},
        ]
        # Set the inputs
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        # Get the output
        outputs = model.generate(**inputs, max_new_tokens=40)
        # Retrieve the response
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        # Display both prompts
        print(f"Original prompt: {prompt}.\nGenerated prompt: {response}")

        # Display elpased time
        print(f"Elapsed time: {round(time.time()-start_time,2)} seconds.")

        # Free memory
        del response, outputs, inputs, messages, model, tokenizer
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
        # Return failure 
        return -1

def test_phi(prompt:str, access_token:str, negative_prompt:str, GPU="0"):
    """
    Tests the Phi-4 model. 
    Steps : 
    1. Loads the model and tokenizer
    2. Prepares the inputs. 
    3. Performs inference (with max new tokens = 400).
    4. Retrieves the output and displays previous and currrent prompt.
    
    (str) prompt: The prompt for image generation.
    (str) access_token: Access token for HuggingFace.
    (str) negative_prompt: The negative prompt.
    """

    # Initialize model name 
    model_name = "microsoft/Phi-4-reasoning-plus"

    # Initialize the start time
    start_time = time.time()

    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # Initialize the input
        messages = [
            {"role": "system", "content": "/nothink"},
            {"role": "user","content": prompt + ". For instance, in 'I am the best swimmer, you would rewrite as: I am th best swmmer.' "}
            ]
        # Setup the inputs
        inputs = tokenizer.apply_chat_template(messages, 
                                               tokenize=True, 
                                               add_generation_prompt=True, 
                                               return_tensors="pt")
        
        # Run inference without gradients
        with torch.no_grad():
            # Retrieve the outputs
            outputs = model.generate(**inputs, 
                                     max_new_tokens=100)
            # Retrieve the response
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

        # Display both prompts
        print(f"Original prompt: {prompt}.\nGenerated prompt: {response}")

        # Display elpased time
        print(f"Elapsed time: {round(time.time()-start_time,2)} seconds.")

        # Free memory
        del response, outputs, inputs, messages, model, tokenizer
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
        # Return failure 
        return -1
    

def test_models(access_token,models=[""], prompt="", GPU="0"):
    """
    Loops through the list of models and test each independently. 
    """

    # Initialize variables
    negative_prompt = "Do not explain. ONLY generate the answer."

    # Loop through the models
    for model in models: 
        # Phi-4
        """if model=="microsoft/Phi-4-reasoning-plus":
            # Test the stable diffusion model
            if test_phi(prompt,access_token,negative_prompt,GPU)==-1:
                # Failure 
                print("\n\nPhi-4: KO\n\n")
            else:
                # Success
                print("\n\nPhi-4: OK\n\n")"""
        # Deepseek
        """elif model=="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":
            # Test the stable diffusion model
            if test_deepseek(prompt,access_token,negative_prompt,GPU)==-1:
                # Failure 
                print("\n\nDeepseek: KO\n\n")
            else:
                # Success
                print("\n\nDeepseek: OK\n\n")"""
        # Mindlink 32B
        if model=="Skywork/MindLink-32B-0801":
            # Test the stable diffusion model
            if test_mindlink(prompt,access_token,negative_prompt,GPU)==-1:
                # Failure 
                print("\n\nSkyword: KO\n\n")
            else:
                # Success
                print("\n\nSkyword: OK\n\n")
        # Qwen 30b
        """elif model=="Qwen/Qwen3-30B-A3B-Instruct-2507": 
            # Test the stable cascade model
            if test_qwen(prompt, access_token,negative_prompt,GPU)==-1:
                # Failure
                print("\n\nQwen : KO\n\n")
            else:
                # Success
                print("\n\nQwen: OK\n\n")"""
        

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
    PROMPT = 'Insert few typos in this text : "A playful scene showing a bright blue elephant standing proudly beside a compact purple car. In the foreground, there is a small cluster of large, pink apples positioned between the car and the elephant; the apples are bigger than the little car but remain noticeably smaller than the towering blue elephant."'
    # Set the list of models
    MODELS = [
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Skywork/MindLink-32B-0801", 
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "microsoft/Phi-4-reasoning-plus"
        ]
    # Load the models 
    test_models(access_token, MODELS, PROMPT, GPU)


if __name__=="__main__":
    # Load the main
    main()