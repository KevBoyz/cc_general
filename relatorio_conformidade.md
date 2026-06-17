# Relatório de Conformidade — analysis.ipynb vs AV3-CD-IA.pdf

Verificação célula por célula do notebook contra o checklist do documento (Questões 1-6), desconsiderando completude de texto explicativo/markdown — avaliando apenas se o passo técnico/código exigido existe e está correto.

## Questão 1 – Análise Exploratória dos Dados — ✅ Sólida

Descrição geral, problemas nos dados (duplicatas, negativos, frequências fora de [0,1], decomposição de compras, nulos), estatísticas descritivas e visualizações (histogramas, skewness, boxplot, pairplot, KDE) — todos os 4 itens pedidos estão implementados.

## Questão 2 – Análise Visual com Dois Atributos — ⚠️ Parcial

- Dispersão entre atributos: feito (e excedido — pairplot + grid de 12 KDEs em vez de só 2 atributos).
- **Faltando**: item 4 do PDF ("colorir os pontos pela variável categórica de referência") — não existe nenhuma coloração por variável categórica aqui, porque o dataset não tem uma variável categórica de referência real (só atributos numéricos). Limitação do dataset, não um bug, mas item do checklist sem resposta.

## Questão 3 – K-Means e Escolha do Melhor K — ⚠️ Parcial

- Seleção de atributos, pré-processamento, teste de K, cotovelo, Silhouette: presentes.
- **Gap 1**: no KMeans "sem normalização" (célula `4a88a9ca`), `n_clusters=2` é hardcoded direto — não há código que derive esse valor do elbow/silhouette calculado na célula anterior (diferente da versão normalizada, que usa `best_k_by_normalizer` via argmax explícito). O "melhor K" do baseline não é rastreável ao resultado do cotovelo/silhouette.
- **Gap 2 (estrutural)**: a "tabela cruzada com a variável categórica real" pedida em Q3 usa, na prática, uma versão discretizada do próprio `PRC_FULL_PAYMENT` — um dos atributos usados para clusterizar. É circular (comparar o cluster com uma versão binarizada de um input dele mesmo), não uma variável de referência externa real. Decorre da ausência de variável categórica no dataset.

## Questão 4 – Comparação K-Means vs DBSCAN — ❌ Gap real (maior risco)

- eps/min_samples, aplicação do DBSCAN, contagem de clusters/ruído, visualização: tudo presente e corrigido (k-distance plot, filtro de ruído, scatter+crosstab espelhando o KMeans).
- **Faltando por completo**: não existe nenhuma célula que junte os resultados de KMeans e DBSCAN lado a lado (quantidade de clusters, ruído, formato, sensibilidade à escala, relação com a "variável categórica"). Os dois algoritmos foram implementados em paralelo, mas nunca comparados entre si em uma tabela/gráfico único. Item de maior peso (2,0 pts) com a maior lacuna.

## Questão 5 – Impacto da Normalização — ⚠️ Parcial

Z-score, Min-Max, Log (+ Robust extra) testados para os dois algoritmos, com seções "sem"/"com" separadas e gráficos comparáveis. Falta uma célula que consolide isso numa comparação explícita (ex: tabela "sem-norm vs melhor-com-norm" lado a lado) — os números existem espalhados em dataframes diferentes, falta juntar.

## Questão 6 – Hierárquica e Dendrograma — ❌ Maior gap de todos

- Seleção de atributos, geração de dendrograma, múltiplos linkages: presentes (gráficos corrigidos recentemente — truncamento para legibilidade).
- **Gap 1**: TODO explícito (`c1731654`) pedindo normalizar antes de aplicar (sugerindo RobustScaler) — nunca foi implementado. `compare_linkages` e o loop de dendrogramas rodam direto em `X_train_cut` sem normalização, com o mesmo problema de escala já corrigido pro DBSCAN (distâncias dominadas por BALANCE/PURCHASES/CASH_ADVANCE).
- **Gap 2**: item "indicar quantidade de clusters com base no corte do dendrograma" (pedido explícito do PDF) — TODO (`d03aa3a9`) nunca implementado. Não há corte no eixo Y nem `fcluster` calculado.
- **Gap 3**: "comparar número de clusters sugerido com o K do KMeans" — também só ficou como TODO (`a55058d2`), nunca implementado.

## Critérios de Avaliação — risco por item

| Critério | Pontos | Status |
|---|---|---|
| Análise exploratória | 0,5 | ✅ Completo |
| Análise visual com 2 atributos | 0,5 | ⚠️ Falta coloração por variável de referência (limitação do dataset) |
| K-Means + cotovelo + Silhouette + crosstab | 2,0 | ⚠️ K do baseline não rastreável ao cotovelo; crosstab é circular |
| Comparação K-Means vs DBSCAN | 2,0 | ❌ Sem comparação explícita entre os dois — maior risco |
| Impacto da normalização | 1,0 | ⚠️ Building blocks existem, falta consolidar comparação |
| Hierárquica e dendrograma | 1,0 | ❌ Normalização nunca aplicada + corte do dendrograma + comparação com K do KMeans, todos pendentes |
| PCA + classificação (Q7, fora do escopo "até Q6") | 2,0 | Parcial — PCA feito, mas KNN/Árvore e métricas (acurácia, F1, matriz de confusão) ainda não implementados |
| Organização/qualidade do código | 1,0 | Time já reconhece dívida técnica (nota inicial sobre "smells, código repetido") |

## Maiores riscos pra nota, em ordem de impacto

1. **Questão 4** sem comparação explícita KMeans×DBSCAN (2,0 pts)
2. **Questão 6** com três pendências nunca resolvidas — normalização, corte do dendrograma, comparação com K (1,0 pt)
3. **Questão 5** sem consolidação da comparação sem/com normalização (1,0 pt)
