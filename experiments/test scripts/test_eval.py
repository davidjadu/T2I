import argparse



def load_arguments():
    """
    """

    return 1

def initialize_parser():
    """
    """

    # Initializing the parser 
    parser = argparse.ArgumentParser(description="T2 Eval test script.")

    # Model name
    parser.add_argument("--model", help="Name of the model to test.", choices=["", 
                                                                               ""
                                                                               "",
                                                                               "", 
                                                                               "", 
                                                                               ""])

    return 1


def main():
    """
        0. Load a specific model, load the eval skills from ArgParser or whatnot
        1. Loop through the generated images folder
        2. For each image, evaluate specific skills and write model's response into a file
        3. Compute metrics
        4. Write metrics into a file 
    """

    # Parse cli arguments
    parser = argparse.ArgumentParser(description="T2I Eval test script.")

    # 


    

    return 1


if __name__=="__main__":
    main()