# Flower-AWS-FederatedLearning

This repository provides an implementation of federated learning using the Flower framework, deployed on AWS. Federated learning allows multiple devices to collaboratively train a machine learning model without sharing raw data, thus maintaining privacy.

## Overview

Federated learning is a machine learning setting where many clients (devices or edge nodes) collaborate in training a model under the coordination of a central server while keeping the training data decentralized. This repository leverages AWS cloud services to deploy and manage federated learning at scale using the Flower framework.

## Features

- **Federated Learning with Flower**: Uses the Flower federated learning framework to facilitate collaboration between distributed clients without sharing sensitive data.
- **AWS Integration**: Utilizes AWS services for deployment and scaling, ensuring robust and scalable federated learning processes.
- **Privacy-Preserving**: No raw data is shared between clients or with the server, preserving privacy and data security.
- **Scalability**: Suitable for both small and large-scale federated learning tasks.

## Prerequisites

- Python 3.x
- AWS account with the necessary permissions (EC2, S3, etc.)
- Docker (for containerizing clients and server)

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/peii14/Flower-AWS-FederatedLearning.git
cd Flower-AWS-FederatedLearning
```

### **2. Install dependencies**
```bash
pip install -r requirements.txt
```
### **3. Configure AWS**
Make sure you have the AWS CLI installed and configured with your credentials:
```bash
aws configure
```
Ensure that the necessary AWS services are active (EC2, S3, etc.) to run the federated learning infrastructure.
### **4. Set up Docker for clients and server**
Build Docker images for the federated learning clients and server:
```bash
docker build -t flower-client ./client
docker build -t flower-server ./server
```

### **5. Deploy on AWS**
Use AWS services such as EC2 or ECS to deploy the clients and server. The server.py script will coordinate training rounds, and clients will send their locally trained model updates back to the server.

## **Running the Project**
### **1. Start the server**
Run the federated learning server on your AWS instance:
```bash
python server.py
```

### **2. Start the clients**
On separate AWS instances or local machines, run the client script:
```bash
python client.py
```

The clients will train on local data and send their updates to the server.

###  **3. Monitor Training**
The server will aggregate the updates from the clients and update the global model. Training can be monitored through server logs or integrated with AWS CloudWatch for real-time monitoring.

## **AWS Services Used**
• **EC2**: To run the federated learning server and clients.
• **S3**: For storage of model parameters and checkpoints.
• **ECS (optional)**: For scaling the deployment of clients and server.

## **Customization**

You can modify the federated learning model and training logic in the client.py and server.py scripts to suit your specific use case. The Flower framework is flexible and allows you to experiment with various aggregation strategies and model architectures.

**References**
• [Flower Framework Documentation](https://flower.dev/docs/)
• [AWS Documentation](https://aws.amazon.com/documentation/)

**License**
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
