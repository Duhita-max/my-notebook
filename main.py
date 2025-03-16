from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)


model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        
        features = [float(request.form[f]) for f in request.form]
        features = np.array(features).reshape(1, -1)

        
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        features_df = pd.DataFrame(features, columns=feature_names)

        
        features_scaled = scaler.transform(features_df)

        
        prob_disease = model.predict_proba(features_scaled)[0][1]

        
        print("🟢 Disease Probability:", prob_disease)

        
        threshold = 0.5  

        
        if prob_disease > threshold:
            prediction = "Disease Detected"
        else:
            prediction = "No Disease"

        
        print("🟢 Final Prediction:", prediction)

        
        return render_template("index.html", result=prediction)

    return render_template("index.html", result="")

if __name__ == "__main__":
    app.run(debug=True)








