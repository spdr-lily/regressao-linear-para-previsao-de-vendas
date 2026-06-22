# 📊 Previsão de Vendas com Machine Learning

[![Streamlit App](https://img.shields.io/badge/Streamlit-Deploy-green)](https://share.streamlit.io)

## Sobre o Projeto

Este projeto utiliza **Regressão Linear** para prever vendas futuras com base em dados históricos e **Regressão Logística** para detecção de fraudes em transações financeiras.

### Funcionalidades

- **Previsão de Vendas**: Modelo de regressão linear usando meses e eventos promocionais
- **Detecção de Fraudes**: Modelo de regressão logística para classificar transações
- **Visualização Interativa**: Gráficos e métricas em tempo real via Streamlit
- **Previsões em Tempo Real**: Interface para fazer novas previsões

### Métricas dos Modelos

| Modelo | MSE | R² | Acurácia |
|--------|-----|----|----------|
| Vendas | 3048.97 | -1.10 | - |
| Fraudes | - | - | ~50% |

> **Nota:** Os modelos foram treinados com dados sintéticos. O R² negativo indica que o modelo linear não é ideal para estes dados.

## Como Executar

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Acesse Online

[![Deploy com Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## Tecnologias

- Python
- Streamlit
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn

## Licença

Apache License 2.0
