#Trademark Classification REST API
Project Overview
This project implements a RESTful API using Django REST Framework to classify goods and services descriptions into their corresponding trademark classes. The machine learning model used for classification is a fine-tuned BERT model. The API allows users to submit descriptions of their goods or services and receive a predicted trademark class in response.

Features
Prediction: Given a description of goods/services, the API returns the predicted trademark class.
User-based Request Limiting: Each user can only make a maximum of 5 API requests. If the limit is exceeded, the API returns an HTTP 429 status code.
Inference Time Logging: The time taken to make each prediction is logged and included in the API response.
API Logging: All API calls and significant events (such as errors and user request counts) are logged using Python's logging framework.
Dockerized Deployment: The entire application is containerized using Docker, making it easy to deploy.

#Technologies Used
Django: Web framework for developing the API.
Django REST Framework: For building RESTful APIs.
PyTorch: For loading and using the BERT model.
Hugging Face Transformers: For accessing the BERT model and tokenizer.
PostgreSQL: Database used to store user request data.
Docker: Containerization of the application.
WandB (Weights & Biases): Used for model tracking and logging during training.

#Project Structure
trademark_api/
│
├── classification/                   # Django app for the classification API
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py                     # Database models
│   ├── serializers.py                # Data serialization
│   ├── utils.py                      # Utility functions including ML model loading and prediction
│   ├── views.py                      # API views
│   └── tests.py                      # Unit tests
│
├── trademark_api/                     # Main Django project settings
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py                       # URL routing
│   ├── wsgi.py
│   └── asgi.py
│
├── Dockerfile                        # Dockerfile for building the container
├── docker-compose.yml                # Docker Compose file
├── requirements.txt                  # Python dependencies
└── manage.py                         # Django's management script

#Setup Instructions
Prerequisites
Ensure you have the following installed:
Python 3.8+
Docker
Docker Compose

#Step 1: Clone the Repository
git clone <repository-url>
cd trademark_api

#Step 2: Set Up the Python Environment
Install dependencies:
pip install -r requirements.txt

#Step 3: Set Up the Database
Run the migrations to create the database schema:
python manage.py migrate

#Step 4: Running the Application
To run the application locally:
python manage.py runserver

#Step 5: Dockerize the Application
Build the Docker image:
docker-compose build
Run the containers:
docker-compose up

#Step 6: Access the API
Once the server is running (either locally or via Docker), you can access the API at:
http://localhost:8000/api/predict/
API Usage
Endpoint: /api/predict/
Method: POST

Description: Predicts the trademark class for the provided description.
Request Body:
user_id: The unique ID of the user making the request.
description: The description of goods/services to classify.

Example: json
{
  "user_id": "user123",
  "description": "Laptop carrying cases"
}

Response:
predicted_class: The predicted trademark class.
inference_time: The time taken to process the request.
Example: json
{
  "predicted_class": "Class 9",
  "inference_time": 0.1234
}

Errors:
429 Too Many Requests: Returned if the user exceeds 5 requests.

#Logging
The application logs important events such as incoming requests, errors, and request limits. The logs are configured in the settings.py file under the LOGGING section.

#Deployment
To deploy the application on a server:

Ensure Docker is installed and set up on the server.
Clone the repository and follow the Dockerization steps above.

You can add unit tests in the classification/tests.py file. To run tests:
python manage.py test
Notes

This documentation provides an overview of the project and detailed instructions on how to set up, run, and use the Trademark Classification API. It should serve as a comprehensive guide for developers looking to understand and work with the project.
