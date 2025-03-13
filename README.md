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

# Upload dataset 
(`az ml data create --name mobile-data \
     --path ./data \
     --type uri_folder \
     --version 1 \
     --workspace-name mlops1 \
     --resource-group mlops`)

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
cd deploy/output
az ml job download --name mango_gyro_m6hsmkt41z --resource-group mlops --workspace-name mlops1


# Register model
cd artifacts/outputs
az ml model create --name "mobile_price_predictor" --path ./mobile_price_predictor.pkl --resource-group mlops --workspace-name mlops1

# Create the endpoint
cd ../../..
az ml online-endpoint create --file endpoint.yml --resource-group mlops --workspace-name mlops1

# Create deployment---> getting error at this point
az ml online-deployment create --file deployment.yml --resource-group mlops --workspace-name mlops1 --debug

# create the endpoint
az ml online-endpoint create --file endpoint.yml --resource-group mlops --workspace-name mlops1

# create deployment 
az ml online-deployment create --file deployment.yml --resource-group mlops --workspace-name mlops1 --debug

# get the endpoint
az ml online-endpoint show \
    --name mobile-price-endpoint \
    --resource-group mlops \
    --workspace-name mlops1 \
    --query scoring_uri \
    --output tsv

# get the primary key
az ml online-endpoint get-credentials \
    --name mobile-price-endpoint \
    --resource-group mlops \
    --workspace-name mlops1 \
    --query primaryKey \
    --output tsv

# Now test the rest api with following CURL
curl -X POST \
   -H "Authorization: Bearer "$THE_TOKEN_THAT_WE_GOT_FROM_UPPER_COMAND" \
 -H "Content-Type: application/json" \
 -d '{"data": [[174.0, 6.0, 3600.0, 6.1, 2024]]}' \
 "$THE_ENDPOINT_THAT_WE_GOT_FROM_GET_THE_ENDPOINT_STEP"
