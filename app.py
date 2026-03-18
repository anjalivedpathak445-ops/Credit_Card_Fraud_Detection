from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import random

app = Flask(__name__)

# ✅ Load model
model = joblib.load("credit_card_model.pkl")


# ✅ Home Page
@app.route('/')
def home():
    return render_template('index.html')


# ✅ Sample Data (NO CSV → No crash)
@app.route('/sample')
def sample():
    sample_data = {}

    # Generate random values similar to dataset
    for i in range(1, 29):
        sample_data[f'V{i}'] = round(random.uniform(-5, 5), 4)

    sample_data['Amount'] = round(random.uniform(1, 5000), 2)

    return jsonify(sample_data)


# ✅ Prediction Route
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


# ❌ DO NOT use debug=True on Render
if __name__ == '__main__':
    app.run()