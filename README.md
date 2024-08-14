<h1 align="center">Trademark Classification API</h1>

<p align="center">
  A REST API for classifying trademarks based on the description of goods and services. The API leverages a BERT-based model to predict the most suitable trademark class.
</p>

<h2>Project Overview</h2>

<p>
    This project involves creating a machine learning model using BERT to classify trademarks into appropriate classes based on the description provided by users. The project includes the following components:
</p>

<ul>
    <li>Preprocessing trademark data and training a BERT-based model.</li>
    <li>Building a REST API using Django to serve the classification model.</li>
    <li>Dockerizing the application for deployment.</li>
    <li>Implementing request rate limiting and logging for the API.</li>
</ul>

<h2>Prerequisites</h2>

<ul>
    <li>Python 3.7+</li>
    <li>Pytorch</li>
    <li>Transformers</li>
    <li>Docker</li>
    <li>Django and Django REST Framework</li>
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

<h2>Getting Started</h2>

<h3>1. Data Preprocessing and Model Training</h3>

<pre><code>
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW

# Load and preprocess data
df = pd.read_json("/content/sample_data/idmanual.json")
df = df[df['status'] == 'A']
X = df['description'].values
y = df['class_id'].values

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128

# Create Dataset class
class TrademarkDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create DataLoader
train_dataset = TrademarkDataset(X_train, y_train, tokenizer, max_len)
val_dataset = TrademarkDataset(X_val, y_val, tokenizer, max_len)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Define model
class TrademarkClassifier(nn.Module):
    def __init__(self, n_classes):
        super(TrademarkClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

model = TrademarkClassifier(len(label_encoder.classes_))
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Training setup
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
loss_fn = nn.CrossEntropyLoss().to('cuda' if torch.cuda.is_available() else 'cpu')
</code></pre>

<h3>2. Model Training</h3>

<pre><code>
import wandb
wandb.login()

# Initialize WandB
wandb.init(project="trademark-classification", settings=wandb.Settings(start_method="fork"))
wandb.watch(model, log="all")

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

# Training loop
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 5
best_accuracy = 0

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, None, len(X_train))
    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(model, val_loader, loss_fn, device, len(X_val))
    print(f'Val loss {val_loss} accuracy {val_acc}')

    wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc

wandb.finish()

def predict(text, model, tokenizer, max_len):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)

    return label_encoder.inverse_transform(prediction.cpu().numpy())[0]

# Example prediction
sample_text = "Laptop carrying cases"
predicted_class = predict(sample_text, model, tokenizer, max_len)
print(f'Predicted class: {predicted_class}')
</code></pre>

<h3>3. REST API Implementation</h3>

<p>The REST API is implemented using Django REST Framework. It allows developers to send descriptions of goods & services and receive the predicted trademark class.</p>

<h2>Features</h2>

<ul>
  <li><strong>Prediction</strong>: Submit a description of goods/services to receive a predicted trademark class.</li>
  <li><strong>User-based Request Limiting</strong>: Each user can make up to 5 API requests. Exceeding this limit returns an HTTP 429 status code.</li>
  <li><strong>Inference Time Logging</strong>: The API logs the time taken for each prediction and includes it in the response.</li>
  <li><strong>API Logging</strong>: All API calls and significant events are logged for monitoring and debugging.</li>
  <li><strong>Dockerized Deployment</strong>: The application is containerized using Docker for easy deployment.</li>
</ul>

<h2>WandB Log Metrics</h2>

![image](https://github.com/user-attachments/assets/b08bcf6d-579b-4c04-b0c5-84e00bee9b5c)


![image](https://github.com/user-attachments/assets/6fb06e8a-8898-4fb7-b8bf-bfa67c1775c3)


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

<b>Due to storage constraints I was not able to push all the required files into the repository, please make use of the same [Google Drive](https://drive.google.com/file/d/15X4n2EpdO4TdZiPwQiYk9xKOgJkAoIK-/view?usp=drive_link)</b>

<b>API Request and response</b>
![image](https://github.com/user-attachments/assets/290e9e45-7d6e-4117-97a5-c5bbf9007159)

<b>ERROR: HTTP 429 status code</b>
![image](https://github.com/user-attachments/assets/3d191f15-aabb-40b3-81cf-dbf833cd1f0c)


<b>API Log file</b>
![image](https://github.com/user-attachments/assets/a0da7d16-3fd6-48ee-8d4c-4146569bc976)



