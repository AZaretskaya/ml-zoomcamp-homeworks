import pickle

model_file = 'model1.bin'
dictionary_vectorizer = 'dv.bin'

with open(model_file, 'rb') as f_in_m:
    model = pickle.load(f_in_m)

with open(dictionary_vectorizer, 'rb') as f_in_d:
    dv = pickle.load(f_in_d)

customer_id = 'from_question_3'
customer = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

def predict(customer):

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    give_card = y_pred >= 0.5

    result = {
        'give_card_probability': round(float(y_pred), 3),
        'give_card': bool(give_card)
    }

    return result

print(predict(customer))

if predict(customer)['give_card']:
    print(f'The client {customer_id} will get a credit card.')
else:
    print(f'The client {customer_id} will not get a credit card.')