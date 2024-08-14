<h1 align="center">Trademark Classification REST API</h1>

<p align="center">
  This repository contains a RESTful API built with Django REST Framework that classifies descriptions of goods and services into their corresponding trademark classes using a fine-tuned BERT model.
</p>

<h2>Features</h2>

<ul>
  <li><strong>Prediction</strong>: Submit a description of goods/services to receive a predicted trademark class.</li>
  <li><strong>User-based Request Limiting</strong>: Each user can make up to 5 API requests. Exceeding this limit returns an HTTP 429 status code.</li>
  <li><strong>Inference Time Logging</strong>: The API logs the time taken for each prediction and includes it in the response.</li>
  <li><strong>API Logging</strong>: All API calls and significant events are logged for monitoring and debugging.</li>
  <li><strong>Dockerized Deployment</strong>: The application is containerized using Docker for easy deployment.</li>
</ul>

<h2>Technologies Used</h2>

<ul>
  <li><strong>Django</strong>: Web framework for building the API.</li>
  <li><strong>Django REST Framework</strong>: For creating RESTful APIs.</li>
  <li><strong>PyTorch</strong>: For using the BERT model.</li>
  <li><strong>Hugging Face Transformers</strong>: For accessing and using the BERT model and tokenizer.</li>
  <li><strong>PostgreSQL</strong>: Database to store user request data.</li>
  <li><strong>Docker</strong>: For containerizing the application.</li>
  <li><strong>WandB (Weights & Biases)</strong>: Used during the training phase for model tracking and logging.</li>
</ul>

<h2>Project Structure</h2>

<pre>
trademark_api/
├── classification/                   # Django app for the classification API
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py                     # Database models
│   ├── serializers.py                # Data serialization
│   ├── utils.py                      # Utility functions, including ML model loading and prediction
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
</pre>

<h2>Setup Instructions</h2>

<h3>Prerequisites</h3>

<p>Make sure you have the following installed:</p>
<ul>
  <li><strong>Python 3.8+</strong></li>
  <li><strong>Docker</strong></li>
  <li><strong>Docker Compose</strong></li>
</ul>

<h3>Step 1: Clone the Repository</h3>

<pre>
<code>
git clone &lt;repository-url&gt;
cd trademark_api
</code>
</pre>

<h3>Step 2: Set Up the Python Environment</h3>

<p>1. Create a virtual environment:</p>
<pre>
<code>
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
</code>
</pre>

<p>2. Install the dependencies:</p>
<pre>
<code>
pip install -r requirements.txt
</code>
</pre>

<h3>Step 3: Set Up the Database</h3>

<p>1. Make sure PostgreSQL is running:</p>
<pre>
<code>
docker-compose up -d
</code>
</pre>

<p>2. Run migrations:</p>
<pre>
<code>
python manage.py migrate
</code>
</pre>

<h3>Step 4: Run the Server</h3>

<p>Start the Django development server:</p>
<pre>
<code>
python manage.py runserver
</code>
</pre>

<h3>Step 5: Making API Calls</h3>

<p>Use an API client like Postman or cURL to interact with the API:</p>

<pre>
<code>
POST /api/predict/
{
  "user_id": "unique_user_id",
  "description": "description of goods or services"
}
</code>
</pre>

<h2>Dockerized Deployment</h2>

<p>To deploy the application using Docker:</p>

<pre>
<code>
docker-compose up --build
</code>
</pre>

<h2>API Reference</h2>

<h3>POST /api/predict/</h3>

<p>Request:</p>
<pre>
<code>
{
  "user_id": "unique_user_id",
  "description": "description of goods or services"
}
</code>
</pre>

<p>Response:</p>
<pre>
<code>
{
  "predicted_class": "predicted trademark class",
  "inference_time": "time taken for prediction"
}
</code>
</pre>

<h3>Due to storage constraints I was not able to push all the required files into the repository, please make use of the same in the google drive link below</h3>
<p><a href="https://drive.google.com/file/d/15X4n2EpdO4TdZiPwQiYk9xKOgJkAoIK-/view?usp=drive_link", target="_blank"></a></p>
