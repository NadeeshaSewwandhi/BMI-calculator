from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Health advice dictionary
advice = {
    "Underweight": [
        "Increase calorie intake with healthy foods.",
        "Do strength training to gain muscle.",
        "Consult a dietitian."
    ],
    "Normal weight": [
        "Maintain a balanced diet.",
        "Exercise regularly.",
        "Stay hydrated and get enough sleep."
    ],
    "Overweight": [
        "Reduce sugar and refined carbs.",
        "Do at least 30 minutes of daily exercise.",
        "Track your calories and portion sizes."
    ],
    "Obese": [
        "Follow a structured weight-loss plan.",
        "Seek professional medical guidance.",
        "Include more fiber and protein in meals.",
        "Avoid sugary drinks and fast food.",
        "Aim for sustainable, gradual weight loss."
    ]
}

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    tips = []
    if request.method == 'POST':
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        features = np.array([[height, weight]])
        prediction = model.predict(features)[0]
        result = f'Your BMI category is: {prediction}'
        tips = advice[prediction]
    return render_template('index.html', result=result, tips=tips)

if __name__ == '__main__':
    app.run(debug=True)
