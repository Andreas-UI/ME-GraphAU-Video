import os
import requests
import gdown  # For Google Drive links

def download_file(url, destination):
    """
    Download a file from the given URL and save it to the specified destination.
    
    Args:
        url (str): The URL of the file to download.
        destination (str): The destination path to save the downloaded file.
    """
    # Check if the URL is a Google Drive link
    if "drive" in url:
        gdown.download(url, destination, quiet=False)
    else:
        # For other links, use requests library
        response = requests.get(url)
        
        # Create the nested directories if they don't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Save the file
        with open(destination, 'wb') as file:
            file.write(response.content)

def download_checkpoints():
    # Example usage
    google_drive_link = "https://drive.google.com/uc?id=1UMnpbj_YKlqHF1m0DHV0KYD3qmcOmeXp"
    other_link = "https://download.pytorch.org/models/resnet50-19c8e357.pth"

    download_file(google_drive_link, os.path.join(os.getcwd(), "MEGraphAU/OpenGraphAU/checkpoints/OpenGprahAU-ResNet50_second_stage.pth"))
    download_file(other_link, os.path.join(os.getcwd(), "MEGraphAU/OpenGraphAU/checkpoints/resnet50-19c8e357.pth"))
