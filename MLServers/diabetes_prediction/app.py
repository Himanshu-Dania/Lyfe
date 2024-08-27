from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from pydantic import BaseModel
import pandas as pd
from flask_cors import CORS
import time

model = pickle.load(open("models/LGBM.pkl", "rb"))

app = Flask(__name__)

CORS(app)


class Scoringitem(BaseModel):
    Age: int
    Gender: bool
    Polyuria: bool
    Polydipsia: bool
    suddenweightloss: bool
    weakness: bool
    Polyphagia: bool
    Genitalthrush: bool
    visualblurring: bool
    Itching: bool
    Irritability: bool
    delayedhealing: bool
    partialparesis: bool
    musclestiffness: bool
    Alopecia: bool
    Obesity: bool


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()
    print("Form Data:", data)

    # Ensure all features are present
    for feature in Scoringitem.__annotations__.keys():
        if feature not in data:
            return render_template(
                "home.html", prediction_text="Missing feature: {}".format(feature)
            )

    # Convert inputs to appropriate types and split the boolean features
    processed_data = {}
    for key, value in data.items():
        if key == "Age":
            processed_data[key] = int(value)
        else:
            bool_value = bool(int(value))
            processed_data[f"{key}_No"] = not bool_value
            processed_data[f"{key}_Yes"] = bool_value

    final_features = pd.DataFrame([processed_data])
    yhat = model.predict(final_features)

    return render_template(
        "home.html", prediction_text="Do you have diabetes: {}".format(yhat[0])
    )


@app.route("/predict_api", methods=["POST"])
def predict_api():
    start = time.time()
    try:
        data = request.get_json(force=True)
        processed_data = {}

        for key, value in data.items():
            if key == "Age":
                processed_data[key] = int(value)
            else:
                bool_value = bool(value)
                processed_data[f"{key}_No"] = not bool_value
                processed_data[f"{key}_Yes"] = bool_value

        # Create a DataFrame
        final_features = pd.DataFrame([processed_data])

        probabilities = model.predict_proba(final_features)
        print(probabilities)
        pred = model.predict(final_features)
        prediction_list = probabilities[0].tolist()
        print(prediction_list)

        for item in prediction_list:
            if not isinstance(item, (int, float)):
                raise TypeError(
                    f"Object of type {type(item).__name__} is not JSON serializable"
                )
        print(time.time() - start)
        return jsonify(
            {"prediction_probability": [prediction_list[pred[0]], int(pred[0])]}
        )

    except Exception as e:
        return jsonify({"error": str(e)})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000, debug=True)
