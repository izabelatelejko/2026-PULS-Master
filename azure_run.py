#!/usr/bin/env python
"""
Azure entrypoint script for running PULS experiments and uploading results to Azure Storage
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import traceback

from azure.storage.blob import BlobServiceClient

from PULS.dataset import Gauss_PULS
from PULS.run import run_experiments


def upload_to_azure(connection_string: str, container_name: str, local_path: str):
    """Upload all files from local_path to Azure Storage"""
    
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    
    local_path = Path(local_path)
    
    if not local_path.exists():
        print(f"Warning: Local path {local_path} does not exist")
        return
    
    uploaded_count = 0
    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_path.parent)
            blob_name = str(relative_path).replace("\\", "/")
            
            print(f"Uploading: {blob_name}")
            with open(file_path, "rb") as data:
                container_client.upload_blob(blob_name, data, overwrite=True)
            uploaded_count += 1
    
    print(f"Successfully uploaded {uploaded_count} files to Azure Storage")


def main():
    """Main execution function"""
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER", "output")
    
    print("=" * 80)
    print("PULS Experiment - Azure Container Instances")
    print("=" * 80)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Container: {container_name}")
    print()
    
    try:
        # Run experiment
        print("Running PULS GaussTest experiment...")
        print("-" * 80)
        
        K = 10
        n = 5000
        label_frequency = 0.5
        dataset_name = "GaussTest"
        
        run_experiments(
            dataset_name=dataset_name, 
            dataset_class=Gauss_PULS, 
            n=n, 
            label_frequency=label_frequency, 
            train_pi_grid=[0.2, 0.4, 0.6, 0.8], 
            test_pi_grid=[0.2, 0.4, 0.6, 0.8], 
            K=K, 
            mean=0.8, 
            verbose=True
        )
        
        print("-" * 80)
        print("Experiment completed successfully!")
        print()
        
        # Upload results to Azure Storage if connection string provided
        if connection_string:
            print("Uploading results to Azure Storage...")
            print("-" * 80)
            upload_to_azure(connection_string, container_name, "./output")
            print("-" * 80)
        else:
            print("Warning: AZURE_STORAGE_CONNECTION_STRING not set. Skipping upload.")
        
        print()
        print("=" * 80)
        print(f"End time: {datetime.now().isoformat()}")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 80)
        print("ERROR: Experiment failed!")
        print("=" * 80)
        print(traceback.format_exc())
        
        if connection_string:
            try:
                print()
                print("Uploading partial results and logs to Azure Storage...")
                upload_to_azure(connection_string, container_name, "./output")
            except Exception as upload_error:
                print(f"Failed to upload error logs: {upload_error}")
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
