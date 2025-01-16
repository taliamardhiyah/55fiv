def predict(model, number):
    prediction = model.predict([[number]])
    return prediction[0]
