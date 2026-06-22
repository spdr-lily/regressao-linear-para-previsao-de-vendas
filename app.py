import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
import sys

st.set_page_config(
    page_title="Previsão de Vendas com Regressão Linear",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_models():
    model_path = 'models/model_data.pkl'
    if not os.path.exists(model_path):
        with st.spinner("Treinando modelos... Aguarde."):
            result = subprocess.run(
                [sys.executable, "train.py"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                st.error(f"Erro ao treinar modelos: {result.stderr}")
                st.stop()
    with open(model_path, 'rb') as f:
        return pickle.load(f)

data = load_models()

st.sidebar.title("Navegação")
pagina = st.sidebar.radio(
    "Ir para:",
    ["Visão Geral", "Previsão de Vendas", "Detecção de Fraudes", "Documentação"]
)

if pagina == "Visão Geral":
    st.title("📊 Projeto de Previsão de Vendas com Machine Learning")
    st.markdown("""
    Este projeto utiliza **Regressão Linear** para prever vendas futuras e **Regressão Logística**
    para detecção de fraudes em transações financeiras.

    ### Modelos disponíveis:
    - **Previsão de Vendas**: Regressão Linear com features 'Meses' e 'Eventos'
    - **Detecção de Fraudes**: Regressão Logística com features 'Valores' e 'Locais'
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MSE (Vendas)", f"{data['mse']:.2f}")
    with col2:
        st.metric("R² (Vendas)", f"{data['r2']:.2f}")
    with col3:
        st.metric("Acurácia (Fraudes)", f"{data['accuracy_fraudes']:.2%}")

    st.subheader("Dados de Vendas")
    st.dataframe(data['df_vendas'], use_container_width=True)

elif pagina == "Previsão de Vendas":
    st.title("📈 Previsão de Vendas - Regressão Linear")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Erro Médio Quadrático (MSE)", f"{data['mse']:.2f}")
    with col2:
        st.metric("Coeficiente de Determinação (R²)", f"{data['r2']:.2f}")

    st.subheader("Comparação: Valores Reais vs Previsões")
    fig, ax = plt.subplots(figsize=(10, 6))
    df_res = data['df_resultado']
    ax.scatter(df_res['Meses'], df_res['Reais'], color='blue', label='Dados Reais', s=80)
    ax.plot(df_res['Meses'], df_res['Previsoes'], color='red', marker='o', label='Previsões')
    ax.set_title('Previsão de Vendas - Regressão Linear')
    ax.set_xlabel('Mês')
    ax.set_ylabel('Vendas')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Tabela de Resultados")
    st.dataframe(df_res, use_container_width=True)

    st.subheader("Faça sua própria previsão")
    col1, col2 = st.columns(2)
    with col1:
        mes = st.number_input("Mês", min_value=1, max_value=36, value=25)
    with col2:
        evento = st.selectbox("Evento Promocional", [0, 1],
                              format_func=lambda x: "Sim" if x == 1 else "Não")

    pred = data['model_vendas'].predict([[mes, evento]])[0]
    st.success(f"Previsão de vendas para o mês {mes}: **R$ {pred:.2f}**")

elif pagina == "Detecção de Fraudes":
    st.title("🔍 Detecção de Fraudes - Regressão Logística")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Acurácia", f"{data['accuracy_fraudes']:.2%}")
    with col2:
        cm = data['confusion_matrix']
        precision = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
        st.metric("Precisão (Fraude)", f"{precision:.2%}")
    with col3:
        recall = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
        st.metric("Recall (Fraude)", f"{recall:.2%}")

    st.subheader("Matriz de Confusão")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Não Fraude', 'Fraude'],
                yticklabels=['Não Fraude', 'Fraude'])
    ax.set_ylabel('Valor Real')
    ax.set_xlabel('Valor Previsto')
    ax.set_title('Matriz de Confusão')
    st.pyplot(fig)

    st.subheader("Dados de Transações")
    st.dataframe(data['df_fraudes_original'].head(20), use_container_width=True)

    st.subheader("Classificação de Transação")
    col1, col2 = st.columns(2)
    with col1:
        valor = st.number_input("Valor da Transação", min_value=0.0, value=1000.0)
    with col2:
        local = st.selectbox("Local", ["Local_A", "Local_B", "Local_C"])

    input_data = pd.DataFrame({
        'Valores': [valor],
        'Locais_Local_A': [1 if local == 'Local_A' else 0],
        'Locais_Local_B': [1 if local == 'Local_B' else 0],
        'Locais_Local_C': [1 if local == 'Local_C' else 0]
    })

    pred_fraude = data['model_fraudes'].predict(input_data)[0]
    prob_fraude = data['model_fraudes'].predict_proba(input_data)[0][1]

    if pred_fraude == 1:
        st.error(f"🚨 **FRAUDE DETECTADA** - Probabilidade: {prob_fraude:.2%}")
    else:
        st.success(f"✅ Transação normal - Probabilidade de fraude: {prob_fraude:.2%}")

elif pagina == "Documentação":
    st.title("📚 Documentação")
    with open("DOCUMENTACAO.md", "r", encoding="utf-8") as f:
        doc_content = f.read()
    st.markdown(doc_content)
