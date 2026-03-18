from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pandas as pd
import random

app = Flask(__name__)

# Load model
model = joblib.load("credit_card_model.pkl")

# Load dataset
data = pd.read_csv("creditcard.csv")


@app.route('/')
def home():
    return render_template('index.html')


# 🔥 Auto-fill route
@app.route('/sample')
def sample():
    # Get random row
    row = data.sample(1).iloc[0]

    sample_data = {}

    # V1 to V28
    for i in range(1, 29):
        sample_data[f'V{i}'] = float(row[f'V{i}'])

    sample_data['Amount'] = float(row['Amount'])

    return jsonify(sample_data)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = []

        for i in range(1, 29):
            features.append(float(request.form[f'V{i}']))

        features.append(float(request.form['Amount']))

        features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        normal_prob = round(probabilities[0] * 100, 2)
        fraud_prob = round(probabilities[1] * 100, 2)

        if prediction == 1:
            result_text = "Fraud Transaction Detected!"
            css_class = "fraud"
        else:
            result_text = "Normal Transaction"
            css_class = "normal"

        return render_template("result.html",
                               prediction=result_text,
                               css_class=css_class,
                               normal_prob=normal_prob,
                               fraud_prob=fraud_prob)

    except Exception as e:
        return render_template("result.html",
                               prediction="Error: " + str(e),
                               css_class="fraud",
                               normal_prob=0,
                               fraud_prob=0)


if __name__ == '__main__':
    app.run(debug=True)