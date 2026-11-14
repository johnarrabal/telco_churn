# Guia de estudo e apresentação

Material de apoio para explicar o **Projeto Integrador** sem precisar ter o código aberto na frente. Leia com calma antes da banca. Os números abaixo são os resultados reais obtidos neste trabalho, com `random_state=42`.

---

## 1. Em uma frase, o que foi feito?

Foi construído um modelo de Machine Learning que, a partir de dados históricos de clientes de telecomunicações, estima a **probabilidade de um cliente cancelar o contrato** (*churn*) e apresenta esse risco em uma aplicação web simples, para apoiar — e não substituir — a análise humana da área de retenção.

---

## 2. O problema, a hipótese e o objetivo

**Problema de pesquisa.** É possível usar dados históricos de clientes de uma operadora para estimar a probabilidade de *churn* de novos clientes?

**Hipótese.** Características como tempo de permanência, tipo de contrato, serviços utilizados, forma de pagamento e valores cobrados apresentam padrões que algoritmos de classificação conseguem aproveitar para estimar esse risco.

**Objetivo geral.** Desenvolver, em Python, uma solução que analise o perfil do cliente, estime a probabilidade de cancelamento e disponibilize o modelo em uma aplicação web.

O escopo é um **MVP**: a pessoa informa os dados de um cliente, o sistema processa esses dados no mesmo *pipeline* usado no treino e devolve probabilidade + faixa de risco + uma interpretação. Fora do escopo (de propósito, para proteger o prazo): login, banco de dados, API, aplicativo mobile, CRM, *deep learning* e *dashboard* corporativo.

---

## 3. Os dados, em números

- **7.043 clientes** e **21 variáveis**.
- A variável que queremos prever é **`Churn`**: *Yes* (cancelou) ou *No* (permaneceu). No modelo, isso vira 1 e 0.
- **1.869 clientes cancelaram (26,54%)** e **5.174 permaneceram (73,46%)**. A base é desbalanceada: a classe que mais nos interessa é minoria.
- O campo `customerID` só identifica o cliente e **não entra no modelo**. Um código pessoal não descreve um padrão generalizável.
- Variáveis numéricas de fato: `tenure` (meses de casa), `MonthlyCharges` (valor mensal) e, depois da limpeza, `TotalCharges` (valor acumulado). `SeniorCitizen` aparece como 0 ou 1; é um indicador (idoso ou não), tratado no *pipeline* como categórica.
- As demais são categóricas: gênero, parceiro, dependentes, telefone, internet, adicionais (segurança, backup, proteção, suporte, *streaming*), tipo de contrato, fatura digital e forma de pagamento.

Se o professor perguntar “quantos clientes e quantas variáveis?”, a resposta direta é: **7.043 e 21**.

---

## 4. A sujeira que havia nos dados (e o que foi feito)

À primeira vista, o `info()` não mostra valores ausentes. Isso é uma armadilha clássica. A coluna `TotalCharges` veio como **texto**, e **11 registros** estavam em branco (só espaço). Depois de `pd.to_numeric(..., errors="coerce")`, esses 11 viraram `NaN`.

O que há de especial nessas 11 linhas?

- Todas têm **`tenure = 0`** (cliente com zero meses de casa).
- Todas têm **`Churn = No`**.
- São apenas **0,16%** da base (11 em 7.043).

A interpretação é operacional: se a pessoa acabou de chegar, ainda não existe valor acumulado de fatura. Por isso o tratamento escolhido foi **preencher com zero**, e não apagar as linhas nem inventar a mediana (a mediana de `TotalCharges` seria típica de quem já tem muitos meses de conta, o que seria incoerente com `tenure = 0`).

Frase pronta: *“Não apaguei os 11 registros sem contar. Contei, entendi o motivo e preenchi com zero, porque são clientes novos, ainda sem faturamento acumulado.”*

---

## 5. O que a análise exploratória mostrou

Foram feitas seis análises, de propósito. Não é quantidade de gráfico que vale, é gráfico com leitura.

1. **Contrato.** Quem está no contrato **mês a mês** tem **42,7%** de *churn*. No contrato de um ano, **11,3%**. No de dois anos, só **2,8%**. É o recorte mais forte da base.
2. **Tempo de casa (`tenure`).** Quem cancelou está, em média, há **18 meses** na empresa; quem ficou, há **37,6 meses**. A mediana é ainda mais contrastante: 10 versus 38 meses. Há muito cancelamento no início do relacionamento.
3. **Valor mensal.** Quem sai paga, em média, **mais**: **US$ 74,44** contra **US$ 61,27**.
4. **Internet.** Fibra óptica: **41,9%** de *churn*. DSL: **19,0%**. Sem internet: **7,4%**.
5. **Suporte técnico.** Sem suporte (tendo internet): **41,6%**. Com suporte: **15,2%**.
6. **Pagamento.** Cheque eletrônico: **45,3%** de *churn*. Os pagamentos automáticos (transferência ou cartão) ficam em torno de **15–17%**.

**Como falar isso na banca, com cuidado.** Esses padrões **sugerem associação**, não causalidade. Clientes satisfeitos podem escolher contrato longo; contrato longo não é, sozinho, a “causa” da permanência. Fibra pode estar ligada a fatura mais alta e a contrato mensal. O modelo aproveita esses padrões para estimar risco; ele não prova que “se eu mudar o contrato, o cliente permanece”.

Um perfil de **maior risco**, olhando a EDA: contrato mensal, pouco tempo de casa, fatura alta, fibra, sem suporte técnico e pagamento com cheque eletrônico.

Um perfil de **menor risco**: contrato de dois anos, muito tempo de casa, fatura mais baixa, serviços de apoio contratados e pagamento automático.

---

## 6. Como o modelo foi preparado e treinado

Fluxo, na ordem em que aconteceu:

1. Limpeza de `TotalCharges`, como acima.
2. Separação: `X` (19 preditoras) e `y` (`Churn` em 0/1). Sem `customerID`.
3. Divisão em **80% treino e 20% teste**, com `random_state=42` (para outra pessoa obter o mesmo resultado) e **`stratify=y`**. A estratificação mantém, no treino e no teste, quase a mesma porcentagem de *churn* da base (cerca de 26,5%). Sem isso, o teste poderia ficar, por acaso, mais fácil ou mais difícil.
4. O teste ficou com **1.409 clientes**, dos quais **374** de fato cancelaram. É nesse conjunto, que o modelo **não viu no treino**, que as métricas foram calculadas.
5. A preparação das variáveis foi colocada em um **`Pipeline`** com **`ColumnTransformer`**:
   - numéricas: imputação de segurança + `StandardScaler` (padronização), importante para a regressão logística, que sente diferença de escala entre, por exemplo, `tenure` e `TotalCharges`;
   - categóricas: imputação pela categoria mais frequente + **`OneHotEncoder`** (cada categoria vira uma coluna 0/1).
6. Esse *pipeline* de preparação é encaixado no mesmo objeto do classificador e **é isso que se grava no disco**. A aplicação não “prepara do seu jeito” e depois manda para um modelo que foi treinado de outro jeito. Esse detalhe é importantíssimo e costuma ser perguntado.

Por que não transformar tudo manualmente no Excel ou com `get_dummies` solto? Porque a aplicação precisaria repetir exatamente os mesmos passos, na mesma ordem, com as mesmas colunas. O *pipeline* evita essa divergência.

---

## 7. Os três modelos e por que `class_weight="balanced"`

O enunciado pede **somente três** algoritmos, com justificativa de escolha, e não uma “copa do mundo” de classificadores:

- **Regressão logística:** *baseline* sólida e relativamente interpretável (pesa as características para estimar uma probabilidade).
- **Árvore de decisão:** regras do tipo “se contrato mensal e tenure baixo, então…”, fáceis de imaginar, mas uma árvore solta e profunda tende a decorar o treino (*overfitting*).
- **Random Forest:** várias árvores combinadas, em geral mais estáveis do que uma árvore única.

A base tem só 26,5% de *churn*. Sem cuidado, o modelo “aprende” a chutar que a maioria permanece e infla a acurácia. Por isso usamos **`class_weight="balanced"`** nos três: a classe minoria passa a ter mais peso no treino, o que costuma **aumentar o recall** de quem cancela. Essa decisão foi tomada em função do objetivo de negócio (não perder de vista quem tende a sair) e confirmada pelas métricas.

---

## 8. O que cada métrica significa (com os números reais)

Imagine o conjunto de teste com 1.409 clientes, 374 dos quais cancelaram.

- **Accuracy (acurácia):** porcentagem de acertos no total. Um modelo ingênuo que dissesse “ninguém sai” já acertaria **73,46%**. Por isso **não olhamos só a acurácia** em *churn*.
- **Precision (precisão) da classe *Churn*:** quando o modelo diz “vai cancelar”, com que frequência ele está certo? Precisão baixa significa muitos **falsos positivos** (apontar quem na verdade ficaria).
- **Recall (sensibilidade) da classe *Churn*:** dos que **realmente** cancelaram, quantos o modelo identificou? Recall baixo significa muitos **falsos negativos** (deixar passar quem vai sair). Neste projeto, esse é o olhar principal.
- **F1-score:** média harmônica de precisão e recall. Equilibra os dois. Foi o segundo critério de escolha.
- **ROC-AUC:** mede a capacidade de **ordenar** clientes por risco (colocar quem cancela com probabilidade maior do que quem permanece). Vale 0,5 no chute aleatório e 1,0 no discriminador perfeito.
- **Matriz de confusão:** tabela do “acerto e erro” em cada classe. É o lugar certo para contar falsos positivos e falsos negativos em gente, não só em porcentagem.

### Tabela real no teste

| Modelo              | Accuracy | Precision | Recall | F1     | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| Regressão Logística |   0,7381 |    0,5043 | 0,7834 | 0,6136 |  0,8415 |
| Árvore de Decisão   |   0,7275 |    0,4867 | 0,4893 | 0,4880 |  0,6517 |
| Random Forest       |   0,7637 |    0,5482 | 0,6230 | 0,5832 |  0,8242 |

### Como ler a regressão logística (modelo final)

- *Recall* **0,7834**: de 374 cancelamentos, o modelo encontrou **293** e deixou passar **81**.
- Precisão **0,5043**: das vezes em que ele apontou *churn*, acertou cerca de metade. Houve **288 falsos positivos**.
- F1 **0,6136** e ROC-AUC **0,8415**: melhor equilíbrio e melhor ordenação de risco entre os três.
- Acurácia **0,7381**: até um pouco menor do que a da *Random Forest* (0,7637). Isso é esperado e aceitável: ganhamos capacidade de achar quem sai, à custa de apontar alguns que ficariam.

### Por que a regressão logística foi a escolhida

Porque o enunciado pede evidência, com prioridade para **recall da classe *Churn*** e **F1**. A logística venceu nesses dois e ainda teve o melhor ROC-AUC. A floresta é honesta e competitiva, mas captura menos cancelamentos. A árvore isolada ficou claramente para trás (ROC-AUC 0,65), sinal de que decorou demais o treino.

O custo do recall alto — os 288 falsos positivos — deve ser assumido com clareza: a ferramenta **não deve disparar desconto automático**. Ela acende um sinal para uma pessoa analisar.

---

## 9. A aplicação

Nome na tela: **TELCO CHURN PREDICTOR**.

A pessoa preenche três blocos:

- **Dados pessoais** (gênero, idoso, parceiro, dependentes);
- **Contrato e serviços** (tempo de casa, telefone, internet e adicionais, tipo de contrato, fatura digital, pagamento);
- **Dados financeiros** (valor mensal e valor total).

Ao clicar em **ANALISAR RISCO DE CHURN**, a aplicação:

1. monta uma linha com as mesmas colunas do treino;
2. passa essa linha pelo *pipeline* salvo em `models/churn_model.joblib`;
3. mostra a **probabilidade em porcentagem** e a faixa **BAIXO / MODERADO / ALTO**.

Faixas (didáticas, não de operadora real):

- 0% a 30%: **BAIXO**
- 30% a 60%: **MODERADO**
- acima de 60%: **ALTO**

A interface ainda exibe um texto de interpretação e um aviso explícito: os limiares são critérios didáticos; a previsão apoia a decisão humana e não recomenda ação comercial automática.

Um detalhe de consistência: se o cliente não tem telefone, “múltiplas linhas” fica como *No phone service*; se não tem internet, os adicionais ficam como *No internet service*. Isso respeita o jeito em que os dados foram gravados no conjunto original.

---

## 10. Limitações que você deve reconhecer (sem o professor precisar puxar)

- A base **não é de uma operadora brasileira atual**. É um conjunto público, clássico, usado para fins acadêmicos.
- **Correlação não é causalidade.**
- O modelo vale na medida em que os dados representam o fenômeno. Se o mercado muda, o modelo pode envelhecer (*drift*).
- A previsão **apoia** a decisão; não a substitui.
- O trabalho é **acadêmico e demonstrativo**. Não é um sistema de CRM, não opera em tempo real e não gera oferta comercial.

Se perguntarem o que viria depois, cite apenas como evolução (não foi implementado): *upload* de lista, *ranking* de risco, *dashboard*, explicabilidade com SHAP, ajuste de limiar, API REST, guardar previsões, integrar com CRM, monitorar *drift* e testar novas bases.

---

## 11. Perguntas prováveis da banca (e respostas curtas)

**Por que não usar só a acurácia?**
Porque a base é desbalanceada. Acertar “o cliente fica” na maioria dos casos gera uma acurácia enganosamente alta e falha exatamente nos cancelamentos, que são o que interessa à retenção.

**O que é recall, no idioma deste problema?**
É a fração dos clientes que **realmente saíram** e que o modelo conseguiu apontar. No modelo final, 78,34% (293 de 374 no teste).

**O que é precisão, no idioma deste problema?**
É a fração de acerto quando o modelo diz “este cliente tende a sair”. Aqui está em 50,43%. Há falsos positivos, e por isso a ferramenta é de apoio, não de disparo automático.

**Por que `stratify=y`?**
Para treino e teste terem quase a mesma mistura de classes. Sem isso, a avaliação poderia parecer melhor ou pior só por azar da divisão.

**Por que salvar o *pipeline* inteiro, e não só o classificador?**
Para a aplicação aplicar **o mesmo** tratamento do treino (imputação, escala, *one-hot*). Se a preparação da tela for diferente da preparação do treino, a previsão deixa de ser válida.

**Por que `class_weight="balanced"`?**
Para compensar os 26,54% de *churn* e melhorar o recall da classe positiva, alinhado ao critério do projeto.

**Por que preencher `TotalCharges` com zero e não apagar?**
Porque os 11 casos têm `tenure = 0`: não houve fatura acumulada. Apagar seria descartar clientes válidos sem necessidade, depois de já sabermos o motivo do branco.

**Por que não usar redes neurais / *deep learning*?**
Ficou fora do escopo de propósito. Com o prazo e o tamanho da base, três modelos clássicos, bem comparados e justificados, atendem ao objetivo. *Deep learning* não era necessário para este MVP.

**O modelo prova que contrato mensal causa *churn*?**
Não. Mostra associação nos dados. Causalidade exigiria outro desenho de estudo.

**O que a área de retenção deveria fazer com um risco ALTO?**
Olhar o caso com atenção, à luz do contexto real do cliente. A aplicação **não** recomenda desconto, brinde ou ligação obrigatória.

**De onde saíram os 30% e 60%?**
São faixas didáticas, definidas para a aplicação acadêmica. Não são regra de uma operadora real. Isso precisa ser dito com todas as letras.

**A Árvore de Decisão não deveria ser mais fácil de explicar?**
Sim, a ideia da árvore é transparente, mas a árvore solta e irrestrita se saiu pior (ROC-AUC 0,65), sinal de sobreajuste. Preferimos um modelo que generaliza melhor no teste. Uma evolução possível seria uma árvore rasa, só para comunicação, ou explicações do tipo SHAP.

**O que é o `random_state=42`?**
É a semente da aleatoriedade. Fixa a divisão treino/teste e o sorteio interno dos algoritmos, para o experimento ser reproduzível.

**SeniorCitizen é numérica ou categórica?**
No arquivo ela é 0/1 (número). Conceitualmente é um sim/não. No *pipeline*, entrou com as categóricas e passou pelo *one-hot*. O importante é ser consistente entre treino e aplicação.

**Como você testou a solução?**
Cinco testes mínimos: cliente aparentemente estável (risco baixo); cliente com perfil de *churn* (risco alto); valores extremos permitidos (por exemplo, `tenure = 0` e `tenure = 72`); preenchimento de todos os campos; e recarga do modelo a partir do disco em um **processo Python novo**, para não depender de um ambiente que já estava aberto.

---

## 12. Roteiro de apresentação (cerca de 5 a 7 minutos)

Use este roteiro como um fio. Não precisa ler palavra por palavra. Se houver demonstração da tela, reserve o minuto 5 para ela.

**0:00–0:45 — Abertura e problema.**
“Empresas de telecomunicações perdem receita quando o cliente cancela. A pergunta deste projeto é: dá para usar o histórico da base para estimar a probabilidade de *churn* de um cliente e apoiar a área de retenção? A hipótese é que contrato, tempo de casa, serviços, forma de pagamento e valores cobrados carregam padrões úteis para isso.”

**0:45–1:45 — Dados e qualidade.**
“Trabalhei com a base Telco Customer Churn da IBM: 7.043 clientes e 21 variáveis. Cerca de 26,5% cancelaram. O identificador do cliente não entra no modelo. Na limpeza, a coluna de valor total veio como texto e tinha 11 linhas em branco, todas com zero meses de casa. Contei os registros, entendi o motivo e preenchi com zero, em vez de apagar sem olhar.”

**1:45–3:00 — O que os dados mostraram.**
“Seis recortes foram suficientes para ver o padrão. Contrato mês a mês: 42,7% de *churn*; contrato de dois anos: 2,8%. Quem sai está, em média, há 18 meses na empresa, contra 38 meses de quem fica. Fatura mensal mais alta, fibra óptica, ausência de suporte técnico e pagamento com cheque eletrônico também aparecem ligados a mais cancelamento. Isso são associações, não causas, mas justificam as variáveis do modelo.”

**3:00–4:30 — Modelagem e escolha.**
“Separei 80% para treino e 20% para teste, com estratificação e semente fixa. A preparação das variáveis ficou em um *pipeline* único, o mesmo que a aplicação usa depois. Comparei três modelos clássicos — regressão logística, árvore de decisão e *random forest* — todos com peso balanceado de classe, porque a base é desbalanceada e eu priorizo o recall de quem cancela. A regressão logística foi a escolhida: recall de 78,34%, melhor F1 (0,6136) e melhor ROC-AUC (0,8415). Ela identifica 293 dos 374 cancelamentos do teste. A floresta tem acurácia um pouco maior, mas perde mais clientes que saem. A árvore sozinha sobreajustou. Assumo o custo: há falsos positivos, por isso a ferramenta não dispara ação comercial sozinha.”

**4:30–6:00 — Aplicação e demonstração.**
“A aplicação se chama TELCO CHURN PREDICTOR. A pessoa informa dados pessoais, contrato, serviços e valores, aperta *Analisar risco de churn* e recebe a probabilidade, a faixa — baixo até 30%, moderado até 60%, alto acima de 60% — e um texto de interpretação. Esses cortes são didáticos, não são regra de operadora. Vou mostrar dois perfis: um cliente antigo, contrato longo e pagamento automático, e um cliente novo, contrato mensal, fibra e cheque eletrônico.”

**6:00–7:00 — Limitações, ética e fechamento.**
“A base não representa uma operadora brasileira atual. Correlação não é causalidade. O modelo apoia a decisão humana e tem finalidade acadêmica. Como evolução, eu pensaria em *ranking* de risco, explicabilidade com SHAP, ajuste de limiar e monitoramento de *drift* — mas isso ficou de fora de propósito, para entregar um MVP consistente. Em resumo: sim, os dados históricos permitem estimar a probabilidade de *churn* com um modelo simples, auditável e utilizável em uma aplicação, desde que o resultado seja lido como apoio, e não como sentença.”

---

## 13. Frases de segurança (vale decorar o sentido)

- “O projeto está concluído quando eu consigo explicar o fluxo inteiro, do dado à tela, sem depender do código aberto.”
- “Eu não apaguei dado sem saber quantas linhas seriam afetadas.”
- “A aplicação e o treino compartilham o mesmo *pipeline*.”
- “Em *churn*, acurácia alta pode esconder um modelo que não vê quem sai.”
- “Limiar de risco aqui é critério didático, não norma de operadora.”
- “A previsão apoia a pessoa responsável; não determina a ação comercial.”

Boa apresentação.
