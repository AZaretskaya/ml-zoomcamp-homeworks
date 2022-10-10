import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model1.bin'
dictionary_vectorizer = 'dv.bin'

with open(model_file, 'rb') as f_in_m:
    model = pickle.load(f_in_m)

with open(dictionary_vectorizer, 'rb') as f_in_d:
    dv = pickle.load(f_in_d)

app = Flask('card')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    give_card = y_pred >= 0.5

    result = {
        'give_card_probability': round(float(y_pred), 3),
        'give_card': bool(give_card)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)



