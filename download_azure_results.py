#!/usr/bin/env python
"""
Script to download results from Azure Storage to your local machine
Usage: python download_azure_results.py
"""

from pathlib import Path
from azure.storage.blob import BlobServiceClient


def download_from_azure(connection_string: str, container_name: str, local_path: str):
    """Download all files from Azure Storage to local_path"""
    
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    
    Path(local_path).mkdir(parents=True, exist_ok=True)
    
    blobs = container_client.list_blobs()
    downloaded_count = 0
    
    for blob in blobs:
        local_file_path = Path(local_path) / blob.name
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading: {blob.name}")
        blob_client = container_client.get_blob_client(blob.name)
        
        with open(local_file_path, "wb") as file_stream:
            download_stream = blob_client.download_blob()
            file_stream.write(download_stream.readall())
        
        downloaded_count += 1
    
    print(f"Successfully downloaded {downloaded_count} files from Azure Storage")


def main():
    """Main download function"""
    
    print("=" * 80)
    print("Download PULS Experiment Results from Azure Storage")
    print("=" * 80)
    print()
    
    connection_string = input("Enter your Azure Storage Connection String:\n> ").strip()
    
    if not connection_string:
        print("Error: Connection string is required!")
        return 1
    
    container_name = input("Enter container name (default: 'output'): ").strip() or "output"
    local_path = input("Enter local download path (default: './output_azure'): ").strip() or "./output_azure"
    
    try:
        print()
        print("Downloading files...")
        print("-" * 80)
        download_from_azure(connection_string, container_name, local_path)
        print("-" * 80)
        print()
        print(f"Files downloaded to: {Path(local_path).absolute()}")
        return 0
        
    except Exception as e:
        print()
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
