## **Projeto de Previsão de Vendas com Machine Learning**

### **1\. Introdução**

Este projeto tem como objetivo construir um modelo de regressão linear para prever vendas futuras com base em dados históricos. A análise foi realizada utilizando a linguagem Python e bibliotecas como Pandas, NumPy e Scikit-learn.

### **2\. Metodologia**

* ### **Criação e Manipulação da Base de Dados:**

  * Foi gerada uma base de dados simulando vendas mensais e eventos promocionais.  
  * A base de dados inclui as colunas: `Meses` (representando os meses), `Vendas` (valores de vendas) e `Eventos` (indicando a ocorrência de eventos promocionais).  
* **Divisão dos Dados:**  
  * Os dados foram divididos em conjuntos de treino e teste para avaliar o desempenho do modelo.  
  * A divisão foi feita utilizando a função `train_test_split` do Scikit-learn, com 30% dos dados reservados para teste e `random_state=42` para garantir a reprodutibilidade.  
* **Construção e Avaliação do Modelo de Regressão Linear:**  
  * Um modelo de regressão linear foi construído utilizando a classe `LinearRegression` do Scikit-learn.  
  * O modelo foi treinado com os dados de treino e avaliado com os dados de teste.  
  * As métricas utilizadas para avaliar o desempenho do modelo foram o Erro Médio Quadrático (MSE) e o Coeficiente de Determinação (R²).  
* **Visualização dos Resultados:**  
  * Os resultados da previsão foram organizados em um DataFrame para facilitar a visualização.  
  * Foi gerado um gráfico de dispersão e linha para comparar as vendas reais com as previsões do modelo.

### **3\. Resultados**

* **Erro Médio Quadrático (MSE):** 3048.97  
* **Coeficiente de Determinação (R²):** \-1.104

O MSE indica a média do quadrado dos erros entre as previsões e os valores reais. Um valor alto de MSE sugere que o modelo tem dificuldade em prever os valores com precisão. O R² mede a proporção da variância na variável dependente que é previsível a partir das variáveis independentes. Um R² negativo indica que o modelo é pior do que prever a média dos dados.

### **4\. Análise**

Os resultados indicam que o modelo de regressão linear não se ajustou bem aos dados. O alto valor de MSE e o R² negativo sugerem que o modelo não é adequado para prever as vendas com base nas variáveis `Meses` e `Eventos`.

### **5\. Conclusão**

O modelo de regressão linear construído não é eficaz para prever as vendas com os dados fornecidos. Pode ser necessário explorar outras abordagens, como a inclusão de mais variáveis preditoras, a utilização de modelos não lineares ou a aplicação de técnicas de engenharia de features para melhorar a precisão das previsões.
