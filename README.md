# Análise de Churn e Retenção de Clientes — Telecom

Análise exploratória, testes estatísticos e modelo preditivo para entender por que clientes cancelam o serviço e identificar, entre os clientes ativos, quais têm maior risco de cancelamento.

## Problema de Negócio

Uma empresa de telecomunicações quer reduzir a perda de clientes (churn) e precisa responder duas perguntas:

1. **Diagnóstica:** por que os clientes estão cancelando, e qual o impacto financeiro disso?
2. **Preditiva:** É possível identificar, hoje, quais clientes ativos têm alto risco de cancelar, antes que aconteça?

A primeira pergunta orienta mudanças de política (ex. estrutura de contratos, investimento em suporte). A segunda gera uma lista priorizada de clientes para ação imediata da equipe de retenção.

**Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), 7.043 clientes e 21 colunas originais, cobrindo dados demográficos, serviços contratados, tipo de contrato, forma de pagamento e status de cancelamento.

## Tecnologias Utilizadas

- **Python:** Pandas e NumPy (limpeza e manipulação), Matplotlib e Seaborn (visualização), SciPy (testes estatísticos), Scikit-learn (modelagem), Statsmodels (diagnóstico de multicolinearidade)
- **Jupyter Notebook** (VS Code)
- **Git/GitHub** (versionamento)

## Metodologia

1. Limpeza de dados (correção de tipo, remoção de coluna sem poder preditivo, consistência categórica)
2. Análise exploratória univariada, bivariada e multivariada
3. Testes de hipótese (qui quadrado para variáveis categóricas, Mann Whitney U para numéricas, com verificação de normalidade via Shapiro Wilk)
4. Transformação de dados (encoding e padronização, aplicados após a divisão treino/teste para evitar vazamento de dados)
5. Modelagem preditiva com Regressão Logística, com correção de desbalanceamento de classes (`class_weight='balanced'`)
6. Diagnóstico e correção de multicolinearidade (VIF) para garantir coeficientes interpretáveis

## Principais Insights

### 1. Contrato é o fator dominante de churn, mas tempo de casa tem efeito próprio

Clientes com contrato mensal (Month to month) cancelam a uma taxa de 42,7%, contra 11,2% (um ano) e 2,8% (dois anos), a diferença mais impactante do dataset (qui quadrado, p ≈ 0). O tempo de casa (tenure) também influencia diretamente: a taxa de churn cai de 47,4% no primeiro ano para 6,6% entre 60 e 72 meses. Testamos se esse efeito era apenas reflexo do tipo de contrato (clientes de contrato longo naturalmente têm tenure mais alto), e mesmo isolando só o segmento mensal, o tenure ainda reduz o risco de forma estatisticamente significativa (p ≈ 2×10⁻²⁸).

**Impacto financeiro:** o segmento de contrato mensal representa R$ 257.294,15 em receita mensal recorrente (MRR). Com uma taxa de churn de 42,7% nesse grupo, uma fatia substancial dessa receita está em risco ativo.

### 2. Fiber optic cancela mais que DSL por qualidade do serviço, não por preço

Clientes de internet fibra óptica cancelam significativamente mais que clientes de DSL (41,9% contra 18,9%), mesmo pagando o mesmo valor mensal. A diferença chega a 44 pontos percentuais em faixas de preço equivalentes (qui quadrado, p ≈ 9,57×10⁻¹⁶⁰), o que descarta preço como explicação e aponta para um problema de satisfação ou estabilidade do serviço.

### 3. Electronic check tem um problema próprio, não é sobre pagamento manual em geral

Clientes que pagam por Electronic check cancelam mais mesmo controlando por tipo de contrato (53,7% contra aproximadamente 32 a 34% dos demais métodos, dentro do segmento de contrato mensal). O efeito não é explicado por fricção de pagamento manual de forma geral, já que o Mailed check, também manual, tem taxa baixa, semelhante aos métodos automáticos.

### 4. O modelo preditivo confirma, de forma independente, os mesmos padrões da análise exploratória

Foi treinado um modelo de Regressão Logística capaz de identificar clientes em risco com AUC ROC de 0,84, capturando 79% dos casos reais de churn (recall), contra 56% de um modelo sem ajuste de peso de classe. Na interpretação dos coeficientes, identificamos multicolinearidade significativa entre `monthly_charges`, `total_charges` e as colunas de serviços contratados (VIF acima de 800 na variável mais problemática). Essas variáveis foram removidas em uma segunda versão do modelo, dedicada à interpretação, sem perda relevante de performance (AUC ROC 0,839). Os coeficientes dessa versão confirmam, de forma independente, os mesmos quatro achados acima: contrato e tenure como maiores redutores de risco, fibra óptica e Electronic check como maiores fatores de risco.

## Recomendações Práticas

1. **Concentrar esforço de retenção no primeiro ano de contrato**, sobretudo entre clientes de contrato mensal, com ações como desconto por migração para contrato anual/bienal, contato proativo e onboarding mais forte nos primeiros meses.
2. **Investigar a experiência do serviço de fibra óptica** (estabilidade, suporte técnico), já que o problema não está no preço cobrado.
3. **Investigar a jornada de pagamento por Electronic check**, entendendo o que torna esse método específico um sinal de risco, antes de assumir que a causa é a ausência de automação.
4. **Usar o modelo preditivo para gerar uma lista priorizada de clientes ativos por probabilidade de cancelamento**, permitindo que a equipe de retenção atue antes do cancelamento acontecer, e não depois.

## Como Reproduzir

```bash
git clone <link-do-repositorio>
cd churn-telecom
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Baixe o dataset em [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) e coloque em `data/`. Depois, abra `notebook_eda.ipynb`.
