# Standard library imports
import os
import json
from datetime import datetime

# Third-party imports
import requests
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient


# Load environment variables from .env file
load_dotenv()

# Setup base directory for file paths
base_dir = os.path.dirname(os.path.abspath(__file__))


def send_data_to_server(data):
    """ Sends data to a specified server via POST request. """
    # URL to which the POST request is sent
    url = "https://dotnetappsqldb20240420033344.azurewebsites.net/api/todoes"
    
    # Sending POST request
    requests.post(url, data=data)

def is_internet_connected(url="http://www.google.com"):
    """ Check if there is an internet connection by making a GET request to a given URL. """
    try:
        response = requests.get(url, timeout=100)  # Set a reasonable timeout
        response.raise_for_status()  # Raise an exception for HTTP errors
        return True
    except requests.RequestException as e:
        print(f"Internet connection check failed: {e}")
        return False

def upload_image_to_blob(file_path):
    """ Uploads a file to Azure Blob Storage. """
    try:
        # Create a blob client using the local file name as the name for the blob
        blob_client = container_client.get_blob_client(file_path)

        # Upload the file
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    global container_client
    
    # Azure storage configuration
    connection_string = os.getenv('azure_connection_string')
    container_name = os.getenv('azure_container_name')
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    
    seen_images = set()
    
    # Get the list of images already processed
    try:
        response = requests.get("https://dotnetappsqldb20240420033344.azurewebsites.net/api/test", timeout=10)  # Timeout parameter is optional but recommended
        response.raise_for_status()  # Raise an exception for HTTP error codes
        seen_images = set(x.get('ImageUrl') for x in response.json())
        print("Seen images: ", len(list(seen_images)))
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    

    # Load history data from JSON file
    with open(os.path.join(base_dir, "data", "history.json")) as file:
        history = json.load(file)

    # Process each history entry
    for entry in history:
        if entry["filename"][5:] not in seen_images:
            print(f"Processing {entry['filename']}...")
            cur_entry = {
                "Description": entry["caption"],
                "CreatedDate": datetime.fromisoformat(entry["createdAt"]),
                "ImageURL": entry["filename"],
            }
            
            send_data_to_server(cur_entry)
            upload_image_to_blob(entry["filename"])
        
        # Remove the picture after uploading
        os.remove(entry["filename"])

    # Clear the history in the JSON file
    with open(os.path.join(base_dir, "data", "history.json"), "w") as file:
        json.dump([], file)



if __name__ == "__main__":
    if is_internet_connected():
        main()
    else:
        print("--- No internet connection available. ---")