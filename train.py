import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

os.makedirs('models', exist_ok=True)

np.random.seed(42)
meses = np.arange(1, 25)
vendas = np.random.normal(200, 50, 24)
eventos = np.random.choice([0, 1], size=24)
df_vendas = pd.DataFrame({
    'Meses': meses,
    'Vendas': vendas,
    'Eventos': eventos
})

x_vendas = df_vendas[['Meses', 'Eventos']]
y_vendas = df_vendas['Vendas']
x_train, x_test, y_train, y_test = train_test_split(
    x_vendas, y_vendas, test_size=0.3, random_state=42
)

model_vendas = LinearRegression()
model_vendas.fit(x_train, y_train)
y_pred = model_vendas.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

df_resultado = pd.DataFrame({
    'Meses': x_test['Meses'],
    'Previsoes': y_pred,
    'Reais': y_test
}).sort_values(by='Meses').reset_index(drop=True)

np.random.seed(42)
valores = np.random.normal(1000, 200, 1000)
locais = np.random.choice(['Local_A', 'Local_B', 'Local_C'], size=1000)
fraudes = np.random.choice([0, 1], size=1000)
df_fraudes = pd.DataFrame({
    'Valores': valores,
    'Locais': locais,
    'Fraude': fraudes
})

df_fraudes = pd.get_dummies(df_fraudes, columns=['Locais'])
X_fraudes = df_fraudes.drop('Fraude', axis=1)
y_fraudes = df_fraudes['Fraude']

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_fraudes, y_fraudes, test_size=0.3, random_state=42
)

model_fraudes = LogisticRegression(max_iter=1000)
model_fraudes.fit(X_train_f, y_train_f)
y_pred_f = model_fraudes.predict(X_test_f)

accuracy = accuracy_score(y_test_f, y_pred_f)
cm = confusion_matrix(y_test_f, y_pred_f)
report = classification_report(y_test_f, y_pred_f, output_dict=True)

model_data = {
    'model_vendas': model_vendas,
    'x_test_vendas': x_test,
    'y_test_vendas': y_test,
    'y_pred_vendas': y_pred,
    'df_resultado': df_resultado,
    'mse': mse,
    'r2': r2,
    'model_fraudes': model_fraudes,
    'X_test_fraudes': X_test_f,
    'y_test_fraudes': y_test_f,
    'y_pred_fraudes': y_pred_f,
    'accuracy_fraudes': accuracy,
    'confusion_matrix': cm,
    'classification_report': report,
    'df_vendas': df_vendas,
    'df_fraudes_original': pd.DataFrame({
        'Valores': valores,
        'Locais': locais,
        'Fraude': fraudes
    })
}

with open('models/model_data.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Modelos treinados e salvos com sucesso!")
print(f"Vendas - MSE: {mse:.2f}, R²: {r2:.2f}")
print(f"Fraudes - Acurácia: {accuracy:.2f}")
