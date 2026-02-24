# Azure Deployment Guide for PULS Experiments

This guide shows a setup to run the experiment on Azure Container Instances and download outputs from Azure Storage.

## Prerequisites

- Azure account + active subscription
- Azure CLI
- Docker Desktop

## 1. Create or reuse Azure resources

```powershell
# Create a resource group in Poland Central
az group create --name puls-experiments --location polandcentral

# Set the active subscription
az account set --subscription "<SUBSCRIPTION_ID>"

# Create Azure Container Registry and enable admin user
az acr create --resource-group puls-experiments --name pulsreg123 --sku Basic
az acr update -n pulsreg123 --admin-enabled true

# Create storage account for output files
az storage account create `
  --resource-group puls-experiments `
  --name pulsoutput `
  --location polandcentral `
  --sku Standard_LRS

# Register the Container Instances provider
az provider register --namespace Microsoft.ContainerInstance
```

## 2. Build and push the image

```powershell
# Build the Docker image locally
docker build -t pulsreg123.azurecr.io/puls-experiment:latest .

# Login to ACR and push the image
az acr login --name pulsreg123
docker push pulsreg123.azurecr.io/puls-experiment:latest
```

Verify the image is in ACR:

```powershell
# Verify the image tag exists in ACR
az acr repository show-tags --name pulsreg123 --repository puls-experiment --output table
```

## 3. Run the container on Azure

```powershell
# Create the container group (Linux)
az container create `
  --resource-group puls-experiments `
  --name puls-exp-job-001 `
  --image pulsreg123.azurecr.io/puls-experiment:latest `
  --registry-login-server pulsreg123.azurecr.io `
  --registry-username pulsreg123 `
  --registry-password "<REGISTRY_PASSWORD>" `
  --environment-variables AZURE_STORAGE_CONNECTION_STRING="<STORAGE_CONNECTION_STRING>" `
  --os-type Linux `
  --memory 16 `
  --cpu 1 `
  --restart-policy Never
```

## 4. Check status and logs

```powershell
# Show container state (Running/Succeeded/Failed)
az container show --resource-group puls-experiments --name puls-exp-job-001 --query "instanceView.state" -o tsv

# Show recent logs
az container logs --resource-group puls-experiments --name puls-exp-job-001 --tail 200

# Show last failure reason (if any)
az container show --resource-group puls-experiments --name puls-exp-job-001 --query "instanceView.events[0].message" -o tsv
```

## 5. Download results

```powershell
# Download results from Azure Storage
python download_azure_results.py
```



