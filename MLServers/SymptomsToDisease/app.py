from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

from pydantic import BaseModel, conint
from typing import Dict
import time


model = pickle.load(
    open("models/ExtraTrees_1_000.pkl", "rb")
)

# Define the symptoms
diseases = [
    "(vertigo) Paroymsal  Positional Vertigo",
    "AIDS",
    "Acne",
    "Alcoholic hepatitis",
    "Allergy",
    "Arthritis",
    "Bronchial Asthma",
    "Cervical spondylosis",
    "Chicken pox",
    "Chronic cholestasis",
    "Common Cold",
    "Dengue",
    "Diabetes",
    "Dimorphic hemmorhoids(piles)",
    "Drug Reaction",
    "Fungal infection",
    "GERD",
    "Gastroenteritis",
    "Heart attack",
    "Hepatitis B",
    "Hepatitis C",
    "Hepatitis D",
    "Hepatitis E",
    "Hypertension",
    "Hyperthyroidism",
    "Hypoglycemia",
    "Hypothyroidism",
    "Impetigo",
    "Jaundice",
    "Malaria",
    "Migraine",
    "Osteoarthristis",
    "Paralysis (brain hemorrhage)",
    "Peptic ulcer diseae",
    "Pneumonia",
    "Psoriasis",
    "Tuberculosis",
    "Typhoid",
    "Urinary tract infection",
    "Varicose veins",
    "hepatitis A",
]

symptoms = [
    "itching",
    "skin_rash",
    "nodal_skin_eruptions",
    "continuous_sneezing",
    "shivering",
    "chills",
    "joint_pain",
    "stomach_pain",
    "acidity",
    "ulcers_on_tongue",
    "muscle_wasting",
    "vomiting",
    "burning_micturition",
    "fatigue",
    "weight_gain",
    "anxiety",
    "cold_hands_and_feets",
    "mood_swings",
    "weight_loss",
    "restlessness",
    "lethargy",
    "patches_in_throat",
    "irregular_sugar_level",
    "cough",
    "high_fever",
    "sunken_eyes",
    "breathlessness",
    "sweating",
    "dehydration",
    "indigestion",
    "headache",
    "yellowish_skin",
    "dark_urine",
    "nausea",
    "loss_of_appetite",
    "pain_behind_the_eyes",
    "back_pain",
    "constipation",
    "abdominal_pain",
    "diarrhoea",
    "mild_fever",
    "yellow_urine",
    "yellowing_of_eyes",
    "acute_liver_failure",
    "fluid_overload",
    "swelling_of_stomach",
    "swelled_lymph_nodes",
    "malaise",
    "blurred_and_distorted_vision",
    "phlegm",
    "throat_irritation",
    "redness_of_eyes",
    "sinus_pressure",
    "runny_nose",
    "congestion",
    "chest_pain",
    "weakness_in_limbs",
    "fast_heart_rate",
    "pain_during_bowel_movements",
    "pain_in_anal_region",
    "bloody_stool",
    "irritation_in_anus",
    "neck_pain",
    "dizziness",
    "cramps",
    "bruising",
    "obesity",
    "swollen_legs",
    "swollen_blood_vessels",
    "puffy_face_and_eyes",
    "enlarged_thyroid",
    "brittle_nails",
    "swollen_extremeties",
    "excessive_hunger",
    "extra_marital_contacts",
    "drying_and_tingling_lips",
    "slurred_speech",
    "knee_pain",
    "hip_joint_pain",
    "muscle_weakness",
    "stiff_neck",
    "swelling_joints",
    "movement_stiffness",
    "spinning_movements",
    "loss_of_balance",
    "unsteadiness",
    "weakness_of_one_body_side",
    "loss_of_smell",
    "bladder_discomfort",
    "continuous_feel_of_urine",
    "passage_of_gases",
    "internal_itching",
    "toxic_look_(typhos)",
    "depression",
    "irritability",
    "muscle_pain",
    "altered_sensorium",
    "red_spots_over_body",
    "belly_pain",
    "abnormal_menstruation",
    "watering_from_eyes",
    "increased_appetite",
    "polyuria",
    "family_history",
    "mucoid_sputum",
    "rusty_sputum",
    "lack_of_concentration",
    "visual_disturbances",
    "receiving_blood_transfusion",
    "receiving_unsterile_injections",
    "coma",
    "stomach_bleeding",
    "distention_of_abdomen",
    "history_of_alcohol_consumption",
    "blood_in_sputum",
    "prominent_veins_on_calf",
    "palpitations",
    "painful_walking",
    "pus_filled_pimples",
    "blackheads",
    "scurring",
    "skin_peeling",
    "silver_like_dusting",
    "small_dents_in_nails",
    "inflammatory_nails",
    "blister",
    "red_sore_around_nose",
    "yellow_crust_ooze",
]


print(len(symptoms))
desc = pd.read_csv("datasets/symptom_Description.csv")
prec = pd.read_csv("datasets/symptom_precaution.csv")
app = Flask(__name__)

CORS(app)


class SymptomsSchema(BaseModel):
    itching: int
    skin_rash: int
    nodal_skin_eruptions: int
    continuous_sneezing: int
    shivering: int
    chills: int
    joint_pain: int
    stomach_pain: int
    acidity: int
    ulcers_on_tongue: int
    muscle_wasting: int
    vomiting: int
    burning_micturition: int
    fatigue: int
    weight_gain: int
    anxiety: int
    cold_hands_and_feets: int
    mood_swings: int
    weight_loss: int
    restlessness: int
    lethargy: int
    patches_in_throat: int
    irregular_sugar_level: int
    cough: int
    high_fever: int
    sunken_eyes: int
    breathlessness: int
    sweating: int
    dehydration: int
    indigestion: int
    headache: int
    yellowish_skin: int
    dark_urine: int
    nausea: int
    loss_of_appetite: int
    pain_behind_the_eyes: int
    back_pain: int
    constipation: int
    abdominal_pain: int
    diarrhoea: int
    mild_fever: int
    yellow_urine: int
    yellowing_of_eyes: int
    acute_liver_failure: int
    fluid_overload: int
    swelling_of_stomach: int
    swelled_lymph_nodes: int
    malaise: int
    blurred_and_distorted_vision: int
    phlegm: int
    throat_irritation: int
    redness_of_eyes: int
    sinus_pressure: int
    runny_nose: int
    congestion: int
    chest_pain: int
    weakness_in_limbs: int
    fast_heart_rate: int
    pain_during_bowel_movements: int
    pain_in_anal_region: int
    bloody_stool: int
    irritation_in_anus: int
    neck_pain: int
    dizziness: int
    cramps: int
    bruising: int
    obesity: int
    swollen_legs: int
    swollen_blood_vessels: int
    puffy_face_and_eyes: int
    enlarged_thyroid: int
    brittle_nails: int
    swollen_extremeties: int
    excessive_hunger: int
    extra_marital_contacts: int
    drying_and_tingling_lips: int
    slurred_speech: int
    knee_pain: int
    hip_joint_pain: int
    muscle_weakness: int
    stiff_neck: int
    swelling_joints: int
    movement_stiffness: int
    spinning_movements: int
    loss_of_balance: int
    unsteadiness: int
    weakness_of_one_body_side: int
    loss_of_smell: int
    bladder_discomfort: int
    continuous_feel_of_urine: int
    passage_of_gases: int
    internal_itching: int
    toxic_look_typhos: int
    depression: int
    irritability: int
    muscle_pain: int
    altered_sensorium: int
    red_spots_over_body: int
    belly_pain: int
    abnormal_menstruation: int
    watering_from_eyes: int
    increased_appetite: int
    polyuria: int
    family_history: int
    mucoid_sputum: int
    rusty_sputum: int
    lack_of_concentration: int
    visual_disturbances: int
    receiving_blood_transfusion: int
    receiving_unsterile_injections: int
    coma: int
    stomach_bleeding: int
    distention_of_abdomen: int
    history_of_alcohol_consumption: int
    blood_in_sputum: int
    prominent_veins_on_calf: int
    palpitations: int
    painful_walking: int
    pus_filled_pimples: int
    blackheads: int
    scurring: int
    skin_peeling: int
    silver_like_dusting: int
    small_dents_in_nails: int
    inflammatory_nails: int
    blister: int
    red_sore_around_nose: int
    yellow_crust_ooze: int


@app.route("/", methods=["GET"])
def home():
    return app.send_static_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    start=time.time()
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Create a list of zeros
    features = [0] * len(symptoms)

    # Set the corresponding indices to 1 for the symptoms present in the data
    for symptom in data:
        if symptom in symptoms:
            index = symptoms.index(symptom)
            features[index] = 1

    # Make prediction using the model
    proba = model.predict_proba([features])

    # Get the indices and probabilities of the top 5 classes
    top5_idx = np.argsort(proba[0])[-5:][::-1]
    top5_proba = np.sort(proba[0])[-5:][::-1]

    # Get the names of the top 5 diseases
    top5_diseases = [diseases[i] for i in top5_idx]

    # Prepare the response
    response = []
    for i in range(5):
        disease = top5_diseases[i]
        probability = top5_proba[i]

        # Get the disease description
        disp = (
            desc[desc["Disease"] == disease].values[0][1]
            if disease in desc["Disease"].unique()
            else "No description available"
        )

        # Get the precautions
        precautions = []
        if disease in prec["Disease"].unique():
            c = np.where(prec["Disease"] == disease)[0][0]
            for j in range(1, len(prec.iloc[c])):
                precautions.append(prec.iloc[c, j])

        # Add the disease prediction to the response
        response.append(
            {
                "disease": disease,
                "probability": float(probability),
                "description": disp,
                "precautions": precautions,
            }
        )

    # Send back to the client
    
    print(time.time()-start)
    return jsonify(response)


if __name__ == "__main__":
    app.run(port=7003, debug=True)
