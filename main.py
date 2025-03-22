from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd


app = Flask(__name__)


with open("heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)


with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


columns = ['cp', 'thalach', 'oldpeak', 'ca', 'thal']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        
        features = [float(request.form[col]) for col in columns]
        features = np.array(features).reshape(1, -1)

        
        features_df = pd.DataFrame(features, columns=columns)

        
        features_scaled = scaler.transform(features_df)

        
        probability = model.predict_proba(features_scaled)[0][1]  

        
        if probability > 0.50:
            result = f"🚨 Heart Disease Detected! (Confidence: {probability*100:.2f}%)"
        else:
            result = f"✅ No Heart Disease (Confidence: {probability*100:.2f}%)"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)














