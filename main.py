from data_fetching import get_data
from data_preprocessing import preprocess_data
from model_training import train_model
from prediction import predict

# Mengambil data
data = get_data()

# Preprocessing data
df = preprocess_data(data)

# Melatih model
model = train_model(df)

# Melakukan prediksi
number = 7
prediction = predict(model, number)
print(f"The prediction for number {number} is {prediction}")
