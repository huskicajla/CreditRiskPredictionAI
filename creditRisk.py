# Potrebne biblioteke
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# Callback klasa za praćenje napretka treniranja LSTM modela
class TrainingProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} "
              f"- val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")

# Funkcija za učitavanje i predobradu podataka
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # Obrada nedostajućih ili nevažećih podataka
    if data.isnull().sum().any():
        data = data.dropna()  

    # Izračunavanje indeksa kreditne sposobnosti
    data['debt_to_income_ratio'] = data['loan_amount'] / (data['income'] / 12)

    # Enkodiranje statusa zaposlenja
    data['employment_status'] = LabelEncoder().fit_transform(data['employment_status'])

    # Razdvajanje karakteristika i ciljne varijable
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y

# Funkcija za podjelu podataka na trening i test setove
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Funkcija za skaliranje podataka
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# Funkcija za pripremu podataka za LSTM model
def prepare_lstm_data(X_train, X_test):
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    return X_train_lstm, X_test_lstm

# Funkcija za treniranje tradicionalnih ML modela
def train_traditional_ml_models(X_train, y_train):
    lr_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    return lr_model, rf_model

# Funkcija za treniranje LSTM modela
def train_lstm_model(X_train_lstm, y_train):
    lstm_model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]))),
        Dropout(0.3),
        BatchNormalization(),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer='l2'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=0.001)
    lstm_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    lstm_model.fit(
        X_train_lstm, y_train, 
        epochs=50, 
        batch_size=64, 
        validation_split=0.2,
        callbacks=[TrainingProgress(), early_stopping, lr_scheduler],
        verbose=0
    )
    
    return lstm_model

# Funkcija za predikciju kreditne sposobnosti
def predict_creditworthiness(customer_data, lr_model, rf_model, lstm_model, scaler, X_columns):
    customer_df = pd.DataFrame([customer_data], columns=X_columns)
    customer_scaled = scaler.transform(customer_df)
    customer_lstm = customer_scaled.reshape((customer_scaled.shape[0], 1, customer_scaled.shape[1]))
    lr_prob = lr_model.predict_proba(customer_scaled)[0][1] * 100
    rf_prob = rf_model.predict_proba(customer_scaled)[0][1] * 100
    lstm_prob = lstm_model.predict(customer_lstm)[0][0] * 100
    predictions = {
        "Logistic Regression": lr_prob,
        "Random Forest": rf_prob,
        "LSTM": lstm_prob
    }
    average_risk_probability = sum(predictions.values()) / len(predictions)
    return predictions, average_risk_probability

# Funkcija za evaluaciju modela
def evaluate_models(X_test, y_test, lr_model, rf_model):
    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    print("\nLogistic Regression Report:")
    print(classification_report(y_test, lr_pred))
    print("\nRandom Forest Report:")
    print(classification_report(y_test, rf_pred))

# Učitavanje i obrada podataka
file_path = "credit_risk_data.csv"
X, y = load_and_preprocess_data(file_path)
X_train, X_test, y_train, y_test = split_data(X, y)
X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
X_train_lstm, X_test_lstm = prepare_lstm_data(X_train_scaled, X_test_scaled)
lr_model, rf_model = train_traditional_ml_models(X_train_scaled, y_train)
lstm_model = train_lstm_model(X_train_lstm, y_train)
evaluate_models(X_test_scaled, y_test, lr_model, rf_model)

# Predikcija kreditne sposobnosti za novog klijenta
new_customer = {
    "age": 25,
    "income": 100000,
    "credit_score": 720,
    "loan_amount": 10000,
    "loan_duration": 24,
    "employment_status": 3,
    "debt_to_income_ratio": ((10000 / 24) / (100000 / 12)) * 100
}
result, average_risk_probability = predict_creditworthiness(new_customer, lr_model, rf_model, lstm_model, scaler, X.columns)
print("\nCreditworthiness Predictions for the New Customer (in %):")
for model, probability in result.items():
    print(f" - {model} Prediction: {probability:.2f}% High Risk")
print(f"\nAverage High Risk Probability: {average_risk_probability:.2f}%")
