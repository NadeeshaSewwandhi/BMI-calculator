import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Sample dataset: [Height (m), Weight (kg)]
data = {
    'Height': [1.50, 1.60, 1.65, 1.70, 1.75, 1.80, 1.55, 1.68, 1.72, 1.85],
    'Weight': [45, 55, 60, 70, 80, 95, 50, 72, 78, 105]
}

df = pd.DataFrame(data)
df['BMI'] = df['Weight'] / (df['Height'] ** 2)

# Define BMI categories
def get_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal weight'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

df['Category'] = df['BMI'].apply(get_category)

# Prepare training data
X = df[['Height', 'Weight']]
y = df['Category']

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as model.pkl")
