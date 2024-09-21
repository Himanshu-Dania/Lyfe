import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from pydantic import BaseModel, conint, model_validator, ValidationError
import pandas as pd
from flask_cors import CORS
from catboost import CatBoostClassifier
import time

app = Flask(__name__)

CORS(app)

cb = CatBoostClassifier()
cb.load_model("models/CB")
# lr = pickle.load(open("models/LR", "rb"))
# gb = pickle.load(open("models/GB", "rb"))
# rf = pickle.load(open("models/RF", "rb"))
lgbm = pickle.load(open("models/LGBM", "rb"))
et = pickle.load(open("models/ET", "rb"))
xgb = pickle.load(open("models/XGB", "rb"))

weights = [1 / 4] * 4


class Scoringitem(BaseModel):
    Age: conint(ge=20, le=80)  # Age between 20 and 80
    Sex: conint(ge=0, le=1)  # Sex should be binary: 0 or 1
    RestingBP: conint(ge=0, le=200)  # RestingBP between 0 and 200
    Cholesterol: conint(ge=0, le=603)  # Cholesterol between 0 and 603
    FastingBS: conint(ge=0, le=1)  # FastingBS should be binary: 0 or 1
    MaxHR: conint(ge=60, le=202)  # MaxHR between 60 and 202
    ExerciseAngina: conint(ge=0, le=1)  # ExerciseAngina should be binary: 0 or 1
    ChestPainType_0: conint(ge=0, le=1)  # ChestPainType should be binary: 0 or 1
    ChestPainType_1: conint(ge=0, le=1)
    ChestPainType_2: conint(ge=0, le=1)
    ChestPainType_3: conint(ge=0, le=1)

    @model_validator(mode="after")
    def validate_chest_pain_type(cls, values):
        # Extract chest pain types
        print(type(values))
        chest_pain_types = [
            values.ChestPainType_0,
            values.ChestPainType_1,
            values.ChestPainType_2,
            values.ChestPainType_3,
        ]

        # Print to observe during debugging
        print(f"Final chest_pain_types: {chest_pain_types}")

        # Ensure that exactly one of the chest pain types is 1
        if sum(chest_pain_types) not in [0, 1]:
            raise ValueError(
                "Atmost one of ChestPainType_0, ChestPainType_1, ChestPainType_2, ChestPainType_3 must be 1."
            )
        return values


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict_api", methods=["POST"])
def predict_api():
    start = time.time()
    try:
        # data JSON data from the request
        data = request.get_json(force=True)

        # Validate the input data using Scoringitem
        scoring_item = Scoringitem(**data)

        # Create a DataFrame
        final_features = pd.DataFrame(
            [scoring_item.dict().values()],
            columns=[
                "Age",
                "Sex",
                "RestingBP",
                "Cholesterol",
                "FastingBS",
                "MaxHR",
                "ExerciseAngina",
                "ChestPainType_0",
                "ChestPainType_1",
                "ChestPainType_2",
                "ChestPainType_3",
            ],
        )

        print(final_features)

        # Make predictions
        preds_cb = cb.predict_proba(final_features)
        preds_lgbm = lgbm.predict_proba(final_features)
        preds_et = et.predict_proba(final_features)
        preds_xgb = xgb.predict_proba(final_features)

        preds = np.vstack([preds_cb, preds_lgbm, preds_et, preds_xgb])
        print(preds)
        print(weights)
        final_preds = np.average(preds, axis=0, weights=weights)
        print(f"final {final_preds}")

        prediction_list = (
            [1, final_preds[1]]
            if final_preds[1] > final_preds[0]
            else [0, final_preds[0]]
        )

        # Ensure that the list only contains JSON serializable types
        for item in prediction_list:
            if not isinstance(item, (int, float)):
                raise TypeError(
                    f"Object of type {type(item).__name__} is not JSON serializable"
                )
        print(time.time() - start)

        # Return the prediction in JSON format
        return jsonify({"prediction_probability": prediction_list})

    except ValidationError as e:
        errors = e.errors()
        formatted_errors = [
            {"loc": err["loc"], "msg": err["msg"], "type": err["type"]}
            for err in errors
        ]
        return jsonify({"validation_error": formatted_errors}), 400
    except ValueError as e:
        return jsonify({"validation_error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    try:
        # Extract values from the form data
        data = {
            "Age": float(request.form.get("Age", 0)),
            "Sex": float(request.form.get("Sex", 0)),
            "RestingBP": float(request.form.get("RestingBP", 0)),
            "Cholesterol": float(request.form.get("Cholesterol", 0)),
            "FastingBS": float(request.form.get("FastingBS", 0)),
            "MaxHR": float(request.form.get("MaxHR", 0)),
            "ExerciseAngina": float(request.form.get("ExerciseAngina", 0)),
            "ChestPainType_0": float(request.form.get("ChestPainType_0", 0)),
            "ChestPainType_1": float(request.form.get("ChestPainType_1", 0)),
            "ChestPainType_2": float(request.form.get("ChestPainType_2", 0)),
            "ChestPainType_3": float(request.form.get("ChestPainType_3", 0)),
        }

        # Validate the input data using Scoringitem
        scoring_item = Scoringitem(**data)

        # Create a DataFrame
        final_features = pd.DataFrame(
            [data],
            columns=[
                "Age",
                "Sex",
                "RestingBP",
                "Cholesterol",
                "FastingBS",
                "MaxHR",
                "ExerciseAngina",
                "ChestPainType_0",
                "ChestPainType_1",
                "ChestPainType_2",
                "ChestPainType_3",
            ],
        )

        # Make predictions
        preds_cb = cb.predict_proba(final_features)
        preds_lgbm = lgbm.predict_proba(final_features)
        preds_et = et.predict_proba(final_features)
        preds_xgb = xgb.predict_proba(final_features)

        preds = np.vstack([preds_cb, preds_lgbm, preds_et, preds_xgb])
        final_preds = np.average(preds, axis=0, weights=weights)

        prediction_list = (
            [1, final_preds[1]]
            if final_preds[1] > final_preds[0]
            else [0, final_preds[0]]
        )

        # Ensure that the list only contains JSON serializable types
        for item in prediction_list:
            if not isinstance(item, (int, float)):
                raise TypeError(
                    f"Object of type {type(item).__name__} is not JSON serializable"
                )

        # Render the prediction result
        return render_template(
            "home.html",
            prediction_text="Heart disease: {}".format(
                "Likely" if prediction_list[0] else "Unlikely"
            ),
        )

    except ValidationError as e:
        # Handle validation errors
        validation_errors = [
            {"loc": err["loc"], "msg": err["msg"]} for err in e.errors()
        ]
        return render_template(
            "home.html",
            prediction_text="Validation Error: {}".format(validation_errors),
        )
    except Exception as e:
        # Handle any other exceptions
        return render_template(
            "home.html", prediction_text="An error occurred: {}".format(str(e))
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7001, debug=True)
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from pydantic import BaseModel, conint, model_validator, ValidationError
import pandas as pd
from flask_cors import CORS
from catboost import CatBoostClassifier
import time

app = Flask(__name__)

CORS(app)

cb = CatBoostClassifier()
cb.load_model("models/CB")
# lr = pickle.load(open("models/LR", "rb"))
# gb = pickle.load(open("models/GB", "rb"))
# rf = pickle.load(open("models/RF", "rb"))
lgbm = pickle.load(open("models/LGBM", "rb"))
et = pickle.load(open("models/ET", "rb"))
xgb = pickle.load(open("models/XGB", "rb"))

weights = [1 / 4] * 4


class Scoringitem(BaseModel):
    Age: conint(ge=20, le=80)  # Age between 20 and 80
    Sex: conint(ge=0, le=1)  # Sex should be binary: 0 or 1
    RestingBP: conint(ge=0, le=200)  # RestingBP between 0 and 200
    Cholesterol: conint(ge=0, le=603)  # Cholesterol between 0 and 603
    FastingBS: conint(ge=0, le=1)  # FastingBS should be binary: 0 or 1
    MaxHR: conint(ge=60, le=202)  # MaxHR between 60 and 202
    ExerciseAngina: conint(ge=0, le=1)  # ExerciseAngina should be binary: 0 or 1
    ChestPainType_0: conint(ge=0, le=1)  # ChestPainType should be binary: 0 or 1
    ChestPainType_1: conint(ge=0, le=1)
    ChestPainType_2: conint(ge=0, le=1)
    ChestPainType_3: conint(ge=0, le=1)

    @model_validator(mode="after")
    def validate_chest_pain_type(cls, values):
        # Extract chest pain types
        print(type(values))
        chest_pain_types = [
            values.ChestPainType_0,
            values.ChestPainType_1,
            values.ChestPainType_2,
            values.ChestPainType_3,
        ]

        # Print to observe during debugging
        print(f"Final chest_pain_types: {chest_pain_types}")

        # Ensure that exactly one of the chest pain types is 1
        if sum(chest_pain_types) not in [0, 1]:
            raise ValueError(
                "Atmost one of ChestPainType_0, ChestPainType_1, ChestPainType_2, ChestPainType_3 must be 1."
            )
        return values


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict_api", methods=["POST"])
def predict_api():
    start = time.time()
    try:
        # data JSON data from the request
        data = request.get_json(force=True)

        # Validate the input data using Scoringitem
        scoring_item = Scoringitem(**data)

        # Create a DataFrame
        final_features = pd.DataFrame(
            [scoring_item.dict().values()],
            columns=[
                "Age",
                "Sex",
                "RestingBP",
                "Cholesterol",
                "FastingBS",
                "MaxHR",
                "ExerciseAngina",
                "ChestPainType_0",
                "ChestPainType_1",
                "ChestPainType_2",
                "ChestPainType_3",
            ],
        )

        print(final_features)

        # Make predictions
        preds_cb = cb.predict_proba(final_features)
        preds_lgbm = lgbm.predict_proba(final_features)
        preds_et = et.predict_proba(final_features)
        preds_xgb = xgb.predict_proba(final_features)

        preds = np.vstack([preds_cb, preds_lgbm, preds_et, preds_xgb])
        print(preds)
        print(weights)
        final_preds = np.average(preds, axis=0, weights=weights)
        print(f"final {final_preds}")

        prediction_list = (
            [1, final_preds[1]]
            if final_preds[1] > final_preds[0]
            else [0, final_preds[0]]
        )

        # Ensure that the list only contains JSON serializable types
        for item in prediction_list:
            if not isinstance(item, (int, float)):
                raise TypeError(
                    f"Object of type {type(item).__name__} is not JSON serializable"
                )
        print(time.time() - start)

        # Return the prediction in JSON format
        return jsonify({"prediction_probability": prediction_list})

    except ValidationError as e:
        errors = e.errors()
        formatted_errors = [
            {"loc": err["loc"], "msg": err["msg"], "type": err["type"]}
            for err in errors
        ]
        return jsonify({"validation_error": formatted_errors}), 400
    except ValueError as e:
        return jsonify({"validation_error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    try:
        # Extract values from the form data
        data = {
            "Age": float(request.form.get("Age", 0)),
            "Sex": float(request.form.get("Sex", 0)),
            "RestingBP": float(request.form.get("RestingBP", 0)),
            "Cholesterol": float(request.form.get("Cholesterol", 0)),
            "FastingBS": float(request.form.get("FastingBS", 0)),
            "MaxHR": float(request.form.get("MaxHR", 0)),
            "ExerciseAngina": float(request.form.get("ExerciseAngina", 0)),
            "ChestPainType_0": float(request.form.get("ChestPainType_0", 0)),
            "ChestPainType_1": float(request.form.get("ChestPainType_1", 0)),
            "ChestPainType_2": float(request.form.get("ChestPainType_2", 0)),
            "ChestPainType_3": float(request.form.get("ChestPainType_3", 0)),
        }

        # Validate the input data using Scoringitem
        scoring_item = Scoringitem(**data)

        # Create a DataFrame
        final_features = pd.DataFrame(
            [data],
            columns=[
                "Age",
                "Sex",
                "RestingBP",
                "Cholesterol",
                "FastingBS",
                "MaxHR",
                "ExerciseAngina",
                "ChestPainType_0",
                "ChestPainType_1",
                "ChestPainType_2",
                "ChestPainType_3",
            ],
        )

        # Make predictions
        preds_cb = cb.predict_proba(final_features)
        preds_lgbm = lgbm.predict_proba(final_features)
        preds_et = et.predict_proba(final_features)
        preds_xgb = xgb.predict_proba(final_features)

        preds = np.vstack([preds_cb, preds_lgbm, preds_et, preds_xgb])
        final_preds = np.average(preds, axis=0, weights=weights)

        prediction_list = (
            [1, final_preds[1]]
            if final_preds[1] > final_preds[0]
            else [0, final_preds[0]]
        )

        # Ensure that the list only contains JSON serializable types
        for item in prediction_list:
            if not isinstance(item, (int, float)):
                raise TypeError(
                    f"Object of type {type(item).__name__} is not JSON serializable"
                )

        # Render the prediction result
        return render_template(
            "home.html",
            prediction_text="Heart disease: {}".format(
                "Likely" if prediction_list[0] else "Unlikely"
            ),
        )

    except ValidationError as e:
        # Handle validation errors
        validation_errors = [
            {"loc": err["loc"], "msg": err["msg"]} for err in e.errors()
        ]
        return render_template(
            "home.html",
            prediction_text="Validation Error: {}".format(validation_errors),
        )
    except Exception as e:
        # Handle any other exceptions
        return render_template(
            "home.html", prediction_text="An error occurred: {}".format(str(e))
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7001, debug=True)
