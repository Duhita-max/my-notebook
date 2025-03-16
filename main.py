from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from form
        features = [float(request.form[f]) for f in request.form]
        features = np.array(features).reshape(1, -1)

        # Convert to DataFrame for correct scaling
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        features_df = pd.DataFrame(features, columns=feature_names)

        # Scale the input data
        features_scaled = scaler.transform(features_df)

        # Get the prediction probability for the positive class (index 1)
        prob_disease = model.predict_proba(features_scaled)[0][1]

        # Print prediction probability
        print("🟢 Disease Probability:", prob_disease)

        # Set the threshold for disease detection
        threshold = 0.5  # Threshold for binary classification (default is 0.5)

        # Make the prediction based on the threshold
        if prob_disease > threshold:
            prediction = "Disease Detected"
        else:
            prediction = "No Disease"

        # Print the final prediction
        print("🟢 Final Prediction:", prediction)

        # Return the result to the template
        return render_template("index.html", result=prediction)

    return render_template("index.html", result="")

if __name__ == "__main__":
    app.run(debug=True)








