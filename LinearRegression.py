import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df_house = pd.read_csv('/content/Housing.csv')

print("------- Housing Dataset Head -------")
print(df_house.head())
print("\n------- Housing Dataset Info -------")
df_house.info()

binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df_house[col] = df_house[col].map({'yes': 1, 'no': 0})

furnishing_dummies = pd.get_dummies(df_house['furnishingstatus'], drop_first=True, prefix='furnishing', dtype=int)
df_house = pd.concat([df_house, furnishing_dummies], axis=1)
df_house.drop('furnishingstatus', axis=1, inplace=True)

X = df_house.drop('price', axis=1)
y = df_house['price']

print("\n------- Preprocessed Data Head -------")
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")
multi_model_house = LinearRegression()
multi_model_house.fit(X_train, y_train)
print("\nModel training complete.")

y_pred_house = multi_model_house.predict(X_test)
mae_house = metrics.mean_absolute_error(y_test, y_pred_house)
mse_house = metrics.mean_squared_error(y_test, y_pred_house)
rmse_house = np.sqrt(mse_house)
r2_house = metrics.r2_score(y_test, y_pred_house)

print("\n------- Model Evaluation Metrics -------")
print(f"Mean Absolute Error (MAE): {mae_house:,.2f}")
print(f"Mean Squared Error (MSE): {mse_house:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_house:,.2f}")
print(f"R-squared (R²): {r2_house:.4f}")

coefficients = pd.DataFrame(multi_model_house.coef_, X.columns, columns=['Coefficient'])
print("\n------- Interpretation of Coefficients -------")
print(f"Intercept: {multi_model_house.intercept_:,.2f}")
print(coefficients)

print("\n--- Interpretation Examples ---")
area_coef = coefficients.loc['area']['Coefficient']
aircon_coef = coefficients.loc['airconditioning']['Coefficient']

print(f"For each one-unit increase in 'area', the price is predicted to increase by ${area_coef:.2f}, holding all other features constant.")
print(f"A house with air conditioning is associated with a predicted price increase of ${aircon_coef:.2f} compared to a house without it, holding all other features constant.")
