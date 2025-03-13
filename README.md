# Azure ML MLOps Setup

This repository contains commands and configurations to set up an Azure Machine Learning workspace, manage datasets, compute resources, jobs, models, and endpoints for a machine learning operations (MLOps) pipeline.

## Prerequisites
- Azure CLI installed
- Azure Machine Learning CLI extension (`az extension add -n ml`)
- An Azure subscription
- Resource group named `mlops` created in advance (`az group create --name mlops --location eastus`)

## Setup Instructions

# Create an Azure ML workspace
az ml workspace create --name mlops1 --resource-group mlops --location eastus

# Upload dataset to default datastore
az ml datastore upload -w mlops1 -g mlops \
    -n $(az ml datastore show-default -w mlops1 -g mlops --query name -o tsv) \
    -p ./data \
    -u mobile_data

# Upload dataset from datastore
az ml data create --name mobile-data \
    --path azureml://datastores/workspaceblobstore/paths/mobile_data/mobile_data.csv \
    --type uri_file \
    --version 1 \
    --workspace-name mlops1 \
    --resource-group mlops

# Create compute cluster
az ml compute create --name cpu-cluster \
    --type AmlCompute \
    --min-instances 0 \
    --max-instances 1 \
    --size STANDARD_DS2_V2 \
    --workspace-name mlops1 \
    --resource-group mlops

# Run a job
az ml job create --file job.yml --resource-group mlops --workspace-name mlops1

# Download job outputs
az ml job download --name mango_gyro_m6hsmkt41z --resource-group mlops --workspace-name mlops1

# Register model
az ml model create --name "mobile_price_predictor" --path ./mobile_price_predictor.pkl --resource-group mlops --workspace-name mlops1

# Create the endpoint
az ml online-endpoint create --file endpoint.yml --resource-group mlops --workspace-name mlops1

# Create deployment---> getting error at this point
az ml online-deployment create --file deployment.yml --resource-group mlops --workspace-name mlops1 --debug


