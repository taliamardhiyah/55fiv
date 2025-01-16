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

# Fungsi untuk menampilkan hasil prediksi
def run_predictions(model, periods=20, initial_bet=1000, max_bet=3000000):
    bet = initial_bet
    total_loss = 0
    results = []

    for period in range(1, periods + 1):
        # Prediksi acak untuk demonstrasi (ganti dengan prediksi sebenarnya)
        prediction = predict(model, period % 10)
        result = "big" if period % 2 == 0 else "small"  # ganti dengan hasil sebenarnya

        if prediction == result:
            results.append(f"Periode {period}: {result} {bet} menang")
            bet = initial_bet  # Reset bet jika menang
        else:
            results.append(f"Periode {period}: {result} {bet} kalah")
            total_loss += bet
            bet = min(bet * 2, max_bet)  # Gandakan bet jika kalah

    return results

# Menjalankan prediksi dan menampilkan hasil
results = run_predictions(model)
print("\n".join(results))
