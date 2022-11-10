import pickle

from flask import Flask
from flask import request
from flask import jsonify

import xgboost as xgb

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('heating')

@app.route('/predict', methods=['POST'])
def predict():
    print('Getting house parameters...')
    house = request.get_json()

    X = dv.transform([house])
    features = dv.feature_names_
    dhouse = xgb.DMatrix(X, feature_names=features)

    print('Predicting heating load...')
    y_pred = model.predict(dhouse)
    print(f'Predicted heating load is {y_pred.round(2)}.')


    result = {
        'heating_load': float(y_pred),
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
