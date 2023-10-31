import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = [float(request.form[field]) for field in ['age', 'income', 'family', 'CCavg', 'education', 'mortgage', 'personal_loan', 'securities_account', 'cd_account', 'online']]
    user_input = np.array(user_input).reshape(1, -1)
    prediction = model.predict(user_input)
    result = 'Eligible for Credit Card' if prediction == 1 else 'Not Eligible for Credit Card'
    return render_template('index.html', prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)