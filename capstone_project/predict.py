import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('advertising')

@app.route('/predict', methods=['POST'])
def predict():

    print('Getting user parameters...')
    user = request.get_json()

    X_user = dv.transform([user])

    print('Predicting clicking on an advertisement..')
    y_pred = model.predict_proba(X_user)[0, 1]
    click = y_pred >= 0.5

    print(f'Predicted clicking is "{click}".')
    result = {
        'click_probability': float(y_pred),
        'click_on_ad': bool(click)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
