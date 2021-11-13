import os
from utils import download_file_from_google_drive


def download():
    UNet_weights_file_id = "1HCN5XlD8w3OPEGwAdnU1JD2tqnqBe6cl"
    test_data_file_id = "1xpiRqG1YCLRRyxPp1JFao7HcjoYH4GBu"
    
    
    print("Downloading the test data ...")
    print("--------------------------------------------")
    download_file_from_google_drive(test_data_file_id, "test_data.zip")
    os.system("unzip test_data.zip -d .")
    os.system("rm test_data.zip")
    
    print("Downloading the segmentation model weights ...")
    print("--------------------------------------------")
    if os.path.exists("weights/UNetModule.pt"):
        print("The segmentation model weights is already downloaded.")
    else:
        os.system("mkdir -p weights")
        download_file_from_google_drive(UNet_weights_file_id, "weights/UNetModule.pt")


if __name__ == '__main__':
    download()
