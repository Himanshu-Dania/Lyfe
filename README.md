# Lyfe
Meet Lyfe, your ultimate healthcare companion. This web app is your personal assistant for disease detection (heart, diabetes, skin issues), symptom analysis, and integrated epidemic alerts. The standout features are its AI Chatbot, Epidemic Alert System and Symptoms to Disease Prediction. Lyfe functions as your virtual doctor for real-time health advice and support. Lyfe puts your well-being in your hands, making healthcare accessible, intuitive, and tailored just for you.

## Problem Statement

### Lack of access to healthcare
Many individuals in remote or underserved areas, face challenges in accessing healthcare services. Some individuals are also unable to visit doctors . In such cases, ML can serve as an intermediary by helping users to better manage their health.
### Unawareness to the status of epidemics and other health related events nearby
The average person isn’t aware of the status of epidemics in the country, nor of the precautionary measures to take. This lack of awareness and access to healthcare services can lead to delayed or inadequate treatments.
### 

## Features

### Skin Disease Detection
A deep learning model that can classify skin diseases based on images of skin lesions. The model uses CNN and transfer learning from the VGG16 architecture to extract features from the images and classify them into one of the 8 common skin diseases, such as-

Nevus

Melanoma

Pigmented benign keratosis

Dermatofibroma

Squamous cell carcinoma

Basal cell carcinoma

Vascular lesion

Actinic keratosis

### Symptoms to Disease Prediction

A Light Gradient Boosting Classification Model deployed using Flask that takes the symptoms a user is feeling out of a list of 128 symptoms and classifies them into one of 42 diseases and returns along the description and precautions to take for the disease. The user can also refer to our LyfeBot for further information.

### Gemini Powered Health ChatBot

LyfeBot is a chatbot that can answer health related queries.
It is built on the newly launched Gemini Large Language Model from Google which is a Generative Artificial Intelligence.
It has been trained on health data and is capable of answering user questions about health management and also our app.

### Epidemic Warning System

A warning system created using Google Maps API that alerts users when they walk into a red zone where a contagious disease such as flu, eye flu, chicken pox etc is speading. It also provides precautions to follow to the user.

Doctors are given the privilege to declare a red zone to alert people of their area to prevent spreading rate of a disease.

### Heart Disease Detection
An ensemble of CatBoost models that can classify whether a person has heart disease or not based on biometric and clinical data. The models are hypertuned using Optuna. The model uses features such as-

Age

Sex

Resting Blood Pressure

Cholestrol

Max Heart Rate

Chest Pain induced through exercise

Chest Pain type- 4 types

Typical Angina

Atypical Angina

Non-Anginal Pain

Asymptomatic

### Diabetes Detection

A Light gradient-boosting machine classification model, fine-tuned by Optuna and deployed with Flask, that can diagnose diabetes based on features such as-

Age

Gender

Polyuria - Presence of excessive urination.

Polydipsia - Excessive thirst.

Sudden Weight Loss

Weakness 

Polyphagia - Excessive hunger,

Genital thrush - Irritation around the genital area may be accompanied with or 
without white discharges 

visual blurring

Itching

Irritability

Delayed Healing

Partial Paresis-Partial loss of voluntary movement

Muscle Stiffness

Alopecia - hair loss near the top of head

Obesity

### Blog Site

Blog site powered by MongoDB where doctors are allowed to share their knowledge and expertise with the public through blogs. The platform enables doctors to post blogs on various health topics, such as disease prevention, diagnosis, treatment, research, etc. The platform also allows non-doctors and doctors to interact with each other through comments, questions, and feedback.

## Hurdles we faced

#### Lack of health related datasets
#### Issues in DevOps
#### Large training times for models

## Future Features 


#### Video Call system with doctors
#### Anonymous chat groups for people suffering with common diseases that are considered taboo to facilitate better communication and where people can give tips and precautions to each other for diseases such as STDs, mental health disorders, Anorectal Malformations and Hernias etc
#### Database of doctors and hospitals sorted according to their reviews and specializations to recommend to patients

#### A sentiment analysis tool that monitors the user’s mood and emotions and provides feedback and support accordingly
#### A gamification feature that rewards the user for achieving their health objectives and motivates them to stay on track
#### Improvements to older models

## Installation

### Setting up the website

1. Clone the repository:

   bash
   git clone https://github.com/your-username/your-repository.git
   

2. Navigate to the project directory:

   bash
   cd di
   

3. Install dependencies:

   bash
   npm install
   
   or

   bash
   yarn install
   

### Running the APIs locally
for example to run the diabetes detection model-
1. Go to each flask app and install the requirements and run 
bash
cd diabetes_predicition

2. Then install the particular requirements
bash
pip install -r requirements.txt

3. Then run the flask app
bash
python main.py

4. Then run the next app locally

## Tech Stack

Python
JavaScript
TypeScript
HTML
CSS
MongoDB
Google Cloud
Google Fit API
Google Maps API
Gemini API
TensorFlow
Scikit-Learn
Optuna
Flask



## License

[MIT](https://choosealicense.com/licenses/mit/)
