# Predição de Churn em Clientes de Telecomunicações utilizando Machine Learning e Aplicação Web em Python

Projeto Integrador da formação *Inteligência Artificial e Data Science Aplicadas a Negócios* (CTA Tecnologia Educacional). O trabalho desenvolve um modelo de classificação para estimar a probabilidade de cancelamento (*churn*) de clientes de telecomunicações e o disponibiliza em uma aplicação web simples, construída com Streamlit.

## Problema

É possível utilizar dados históricos de clientes de uma empresa de telecomunicações para desenvolver um modelo de Machine Learning capaz de estimar a probabilidade de *churn* de novos clientes?

Empresas do setor operam em mercados competitivos e dependem da manutenção da base de clientes. O cancelamento de contratos representa perda de receita e pode estar associado a características do serviço, do atendimento, do preço, do tipo de contrato ou do relacionamento com o cliente. A solução aqui apresentada estima o risco de cancelamento para apoiar a análise humana da área de retenção — sem recomendar automaticamente promoções, preços ou decisões comerciais.

## Objetivo

Desenvolver, em Python, uma solução de Machine Learning que analise características de clientes de telecomunicações, estime a probabilidade de *churn* e disponibilize o modelo por meio de uma aplicação web.

Objetivos específicos cumpridos:

1. analisar o conjunto *Telco Customer Churn*;
2. identificar as características disponíveis e a variável-alvo `Churn`;
3. realizar a limpeza e a preparação dos dados;
4. desenvolver a análise exploratória;
5. preparar variáveis categóricas e numéricas para os algoritmos;
6. separar dados de treinamento e de teste;
7. treinar modelos de classificação;
8. comparar o desempenho com métricas adequadas;
9. selecionar o modelo mais adequado;
10. integrar o modelo treinado a uma aplicação Streamlit;
11. permitir que o usuário informe características de um cliente;
12. apresentar a probabilidade e a classificação de risco de *churn*;
13. interpretar os resultados sob a perspectiva de apoio à retenção.

## Dataset

Foi utilizado o conjunto **Telco Customer Churn**, disponibilizado pela IBM.

- Arquivo: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Fonte pública de referência: [repositório IBM *telco-customer-churn-on-icp4d*](https://github.com/IBM/telco-customer-churn-on-icp4d)
- Dimensão: **7.043 clientes** e **21 variáveis**
- Variável-alvo: `Churn`, convertida para forma binária (`No = 0`, `Yes = 1`)
- Identificador `customerID`: não utilizado como variável preditiva
- Distribuição da alvo: 5.174 clientes permaneceram (73,46%) e 1.869 cancelaram (26,54%)

A variável `TotalCharges` chegou ao arquivo como texto e continha 11 registros em branco, todos com `tenure = 0`. Após a conversão com `pd.to_numeric(..., errors="coerce")`, esses valores foram preenchidos com zero, porque correspondem a clientes recém-chegados, ainda sem faturamento acumulado.

## Tecnologias

| Uso                         | Tecnologia        | Versão utilizada |
| --------------------------- | ----------------- | ---------------- |
| Linguagem                   | Python            | 3.12.16          |
| Manipulação de dados        | Pandas            | 3.1.1            |
| Visualização                | Matplotlib        | 3.11.2           |
| Visualização (apoio)        | Seaborn           | 0.13.2           |
| Machine Learning            | Scikit-learn      | 1.9.1            |
| Aplicação web               | Streamlit         | 1.67.0           |
| Serialização do modelo      | Joblib            | 1.6.0            |
| Ambiente                    | Jupyter Notebook  | 1.1.1            |
| Versionamento               | Git               | —                |

As versões efetivamente empregadas estão fixadas em `requirements.txt`. O Streamlit, na documentação atual, recomenda ambiente virtual e instalação via `pip`, com suporte a Python 3.11–3.15. Neste projeto utilizamos **Python 3.12**, conforme a orientação de reduzir risco de incompatibilidade.

## Metodologia

O trabalho seguiu as etapas do enunciado, registradas nos cadernos e na aplicação:

1. **Conhecer os dados.** Leitura do CSV e inspeção com `head`, `shape`, `info`, `describe` e `value_counts` da variável `Churn` (caderno `notebooks/01_exploracao.ipynb`).
2. **Análise exploratória.** Seis recortes foram priorizados, sempre seguidos da pergunta *“O que esse resultado parece indicar?”*: `Contract`, `tenure`, `MonthlyCharges`, `InternetService`, `TechSupport` e `PaymentMethod`, todos cruzados com `Churn`.
3. **Limpeza.** Conversão de `TotalCharges` para numérico e tratamento documentado das 11 linhas afetadas (imputação por zero, justificada pelo `tenure = 0`).
4. **Separação da alvo.** `X` reúne as 19 preditoras; `y` é o `Churn` binário. O `customerID` é descartado.
5. **Treino e teste.** `train_test_split` com `test_size=0.20`, `random_state=42` e `stratify=y`, para preservar a proporção das classes.
6. **Preparação automática.** Um `Pipeline` com `ColumnTransformer` aplica `SimpleImputer` + `StandardScaler` às numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`) e `SimpleImputer` + `OneHotEncoder` às categóricas. O mesmo objeto é reutilizado pela aplicação.
7. **Modelagem.** Comparação de três algoritmos — Regressão Logística, Árvore de Decisão e *Random Forest* — todos com `class_weight="balanced"`, em razão do desbalanceamento (26,54% de *churn*) e da prioridade em elevar o *recall* da classe positiva.
8. **Avaliação e escolha.** Métricas: *accuracy*, *precision*, *recall*, F1-score, ROC-AUC e matriz de confusão. A escolha privilegiou o *recall* da classe *Churn* e o F1-score.
9. **Serialização.** O *pipeline* completo (preparação + modelo escolhido) foi gravado em `models/churn_model.joblib`.
10. **Aplicação e testes.** Interface Streamlit com os campos do cliente, botão de análise, probabilidade, faixa de risco e testes automatizados em `tests/`.

## Modelos comparados

Os três modelos foram treinados no mesmo conjunto de treino e avaliados no mesmo conjunto de teste (1.409 clientes, dos quais 374 com *churn*), com o mesmo *pipeline* de preparação.

## Resultado

Métricas no conjunto de teste:

| Modelo              | Accuracy | Precision | Recall | F1     | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| Regressão Logística |   0,7381 |    0,5043 | 0,7834 | 0,6136 |  0,8415 |
| Árvore de Decisão   |   0,7275 |    0,4867 | 0,4893 | 0,4880 |  0,6517 |
| Random Forest       |   0,7637 |    0,5482 | 0,6230 | 0,5832 |  0,8242 |

**Modelo escolhido:** Regressão Logística com `class_weight="balanced"`.

**Justificativa:** o enunciado pede que a decisão se apoie principalmente no *recall* da classe *Churn* e no F1-score. A regressão logística obteve o melhor *recall* (0,7834: 293 dos 374 cancelamentos do teste foram identificados) e o melhor F1 (0,6136), além do maior ROC-AUC (0,8415). A *Random Forest* teve acurácia e precisão um pouco melhores, mas deixou de capturar mais clientes que de fato saíram. A árvore isolada, sem restrição de profundidade, apresentou ROC-AUC de 0,6517, indício de sobreajuste. A precisão de 0,5043 da regressão implica falsos positivos e reforça o caráter de apoio à decisão humana: a aplicação não deve disparar ofertas automaticamente.

Na aplicação, a probabilidade é traduzida em faixas didáticas:

- 0% a 30%: risco **BAIXO**
- 30% a 60%: risco **MODERADO**
- acima de 60%: risco **ALTO**

Esses limites não foram definidos por uma operadora real.

## Como executar

O projeto foi desenvolvido com **Python 3.12**. Versões 3.11 ou 3.12 são as recomendadas.

### Windows

Abra o *Prompt de Comando* ou o *PowerShell* na pasta do projeto e execute:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para explorar os cadernos:

```bat
jupyter notebook
```

Os cadernos estão em `notebooks\01_exploracao.ipynb` e `notebooks\02_modelagem.ipynb`. Se o Jupyter abrir com o diretório de trabalho em `notebooks\`, os próprios cadernos ajustam o caminho para a raiz do projeto.

Para iniciar a aplicação:

```bat
streamlit run app.py
```

Para rodar os testes:

```bat
pytest -q
```

### Linux e macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
streamlit run app.py
pytest -q
```

O modelo serializado já acompanha o repositório em `models/churn_model.joblib`. Caso deseje retreiná-lo, execute integralmente o caderno `notebooks/02_modelagem.ipynb`.

## Limitações

- O conjunto de dados **não representa necessariamente uma operadora brasileira atual**. Trata-se de uma base pública, clássica na literatura de *churn*, com finalidade acadêmica.
- **Correlação não significa causalidade.** Observar que clientes de contrato mensal cancelam mais não autoriza afirmar, só com estes dados, que o tipo de contrato *cause* o cancelamento.
- O modelo **depende da qualidade e da representatividade dos dados** de treino. Mudanças no mercado, nos produtos ou no perfil da base podem degradar o desempenho.
- A previsão deve **apoiar decisões humanas**, não determinar automaticamente ações comerciais (descontos, retenções ou recusas).
- Os limiares de risco da aplicação são **critérios didáticos**, não regras de uma operadora real.
- O projeto possui **finalidade acadêmica e demonstrativa**. Não há cadastro de usuários, banco próprio, integração com sistemas de telecomunicações nem processamento em tempo real — itens deliberadamente deixados fora do escopo do MVP.

## Possíveis evoluções

As ideias a seguir foram apenas registradas, **sem implementação neste momento**, para preservar o prazo do MVP:

- envio (*upload*) de uma lista de clientes;
- *ranking* dos clientes com maior risco;
- *dashboard* gerencial;
- explicabilidade com SHAP;
- ajuste do limiar (*threshold*) de classificação;
- API REST;
- armazenamento das previsões;
- integração com CRM;
- monitoramento de *drift*;
- utilização de novos conjuntos de dados.

## Estrutura do repositório

```text
telco-churn/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   ├── 01_exploracao.ipynb
│   └── 02_modelagem.ipynb
├── models/
│   └── churn_model.joblib
├── tests/
│   ├── test_predicoes.py
│   └── test_interface_streamlit.py
├── app.py
├── requirements.txt
├── README.md
└── GUIA_DE_ESTUDO.md
```

O arquivo `GUIA_DE_ESTUDO.md` explica o fluxo do projeto em linguagem acessível, com as métricas reais, perguntas frequentes de banca e um roteiro curto de apresentação.
