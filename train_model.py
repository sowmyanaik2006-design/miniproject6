import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Create sample electricity data
data = pd.DataFrame({
    "Units": [
        80, 100, 120, 150, 180, 200, 220, 250, 280, 300,
        90, 110, 140, 160, 190, 210, 230, 260, 290, 320,
        70, 130, 170, 240, 310, 350, 100, 200, 270, 330
    ],
    "Occupants": [
        1, 2, 2, 3, 3, 4, 4, 5, 5, 5,
        1, 2, 3, 3, 4, 4, 4, 5, 5, 6,
        1, 2, 3, 4, 5, 6, 2, 3, 4, 6
    ],
    "Temperature": [
        22, 24, 26, 28, 30, 32, 33, 34, 35, 36,
        23, 25, 27, 29, 31, 32, 33, 34, 35, 36,
        21, 26, 29, 33, 35, 37, 24, 30, 34, 36
    ],
    "Bill": [
        450, 550, 650, 800, 950, 1050, 1150, 1300, 1450, 1550,
        500, 600, 750, 850, 1000, 1100, 1200, 1350, 1500, 1650,
        400, 700, 900, 1250, 1600, 1800, 550, 1050, 1400, 1700
    ]
})

# 2. Define features and target
X = data[["Units", "Occupants", "Temperature"]].values
y = data["Bill"].values

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scale input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Build the neural network
model = Sequential([
    Input(shape=(3,)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1, activation="linear")
])

# 6. Compile the model
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)

# 7. Train the model
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=25,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=4,
    callbacks=[early_stopping],
    verbose=1
)

# 8. Evaluate model
predictions = model.predict(X_test, verbose=0).flatten()

print("\nTraining completed!")
print("Mean Absolute Error:",
      round(mean_absolute_error(y_test, predictions), 2), "rupees")
print("R2 Score:", round(r2_score(y_test, predictions), 2))

# 9. Save model and scaler
model.save("electricity_model.keras")

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("\nSaved electricity_model.keras")
print("Saved scaler.pkl")

# 10. Test a sample prediction
sample = np.array([[200, 4, 32]])
sample_scaled = scaler.transform(sample)
bill = model.predict(sample_scaled, verbose=0)[0][0]

print("Sample predicted monthly bill: Rs.",
      round(float(bill), 2))