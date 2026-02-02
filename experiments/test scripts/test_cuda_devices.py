import torch 


if __name__=="__main__":
    # Display the number of cuda devices
    print(torch.cuda.device_count())
    # Loop through the devices
    for i in range(torch.cuda.device_count()): 
        # Display device name
        print(i, torch.cuda.get_device_name(i))