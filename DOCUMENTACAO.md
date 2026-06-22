# Documentação do Projeto

## Possíveis Erros e Soluções

### 1. Erro: `FileNotFoundError: No such file or directory: 'models/model_data.pkl'`

**Causa:** Os modelos treinados não foram gerados.

**Solução:** Execute o script de treinamento antes de iniciar a aplicação:
```bash
python train.py
```

### 2. Erro: `ModuleNotFoundError: No module named 'streamlit'`

**Causa:** Dependências não instaladas.

**Solução:** Instale todas as dependências:
```bash
pip install -r requirements.txt
```

### 3. Erro: `ValueError: could not convert string to float`

**Causa:** Tentativa de usar dados categóricos não processados no modelo.

**Solução:** Certifique-se de aplicar `pd.get_dummies()` ou `LabelEncoder()` nas variáveis categóricas antes de treinar/predict.

### 4. Erro: R² negativo

**Causa:** O modelo de regressão linear não se ajusta bem aos dados.
- **Solução 1:** Adicione mais features relevantes (sazonalidade, tendência, etc.)
- **Solução 2:** Experimente modelos não lineares (Random Forest, XGBoost)
- **Solução 3:** Aplique engenharia de features (polinômios, interações)

### 5. Erro: `NaN` ou `inf` values durante o treinamento

**Causa:** Dados com valores ausentes ou infinitos.

**Solução:** Limpe os dados antes do treinamento:
```python
df = df.dropna()
df = df.replace([np.inf, -np.inf], np.nan).dropna()
```

### 6. Erro: Baixa acurácia no modelo de fraudes (~50%)

**Causa:** Dados sintéticos aleatórios sem correlação real entre features e target.

**Solução:** Utilize dados reais com correlação comprovada ou gere dados sintéticos com relação definida entre variáveis.

### 7. Erro: `streamlit run app.py` não abre o navegador

**Causa:** Streamlit pode não conseguir abrir o navegador automaticamente.

**Solução:** Acesse manualmente o endereço exibido no terminal (geralmente `http://localhost:8501`).

### 8. Erro: `pickle.UnpicklingError: invalid load key`

**Causa:** Arquivo pickle corrompido ou versão incompatível do Python.

**Solução:** Reexecute `python train.py` para gerar um novo arquivo pickle.

---

## Como Executar o Projeto

### Localmente
```bash
# 1. Clone o repositório
git clone <url-do-repositorio>
cd regressao-linear-para-previsao-de-vendas

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Treine os modelos
python train.py

# 4. Inicie a aplicação web
streamlit run app.py
```

### Deploy no Streamlit Cloud
1. Faça push do código para um repositório GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte seu repositório
4. Configure:
   - Branch: `main`
   - Arquivo principal: `app.py`
5. Clique em "Deploy"

---

## Estrutura do Projeto

```
├── app.py                          # Aplicação Streamlit
├── train.py                        # Treinamento dos modelos
├── requirements.txt                # Dependências
├── DOCUMENTACAO.md                 # Documentação de erros e soluções
├── README.md                       # README principal
├── models/
│   └── model_data.pkl              # Modelos treinados
├── previsao de vendas com regressao linear.ipynb  # Notebook original
├── .gitignore
├── LICENSE
└── AGENTS.md
```
