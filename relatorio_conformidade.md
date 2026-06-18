# Relatório de Conformidade — analysis.ipynb vs AV3-CD-IA.pdf

Verificação do notebook contra o checklist do documento (Questões 1-7). Para os itens de **comparação** explicitamente pedidos pelo PDF (Q4 e Q5), a análise é feita por escrito neste relatório — não é necessário (nem foi pedido) que cada comparação exista como uma célula de código própria; a ausência disso não é tratada como gap.

## Questão 1 – Análise Exploratória dos Dados — ✅ Completa

Descrição geral, problemas nos dados (duplicatas, negativos, frequências fora de [0,1], nulos), estatísticas descritivas e visualizações: histogramas cobrindo as 14 variáveis contínuas, distribuição das 3 variáveis discretas, boxplot combinado + boxplot individual por variável (escala própria), pairplot e KDE. Todos os 4 itens pedidos estão implementados e completos.

## Questão 2 – Análise Visual com Dois Atributos — ⚠️ Parcial (limitação do dataset)

- Dispersão entre atributos: feito de três formas — KDE grid de 12 combinações, e um scatter plot literal (`PRC_FULL_PAYMENT` × `CREDIT_LIMIT`) específico pra atender o pedido textual do PDF.
- **Faltando**: colorir os pontos pela variável categórica de referência (item 4 do PDF) — o dataset não tem uma variável categórica de referência real, só atributos numéricos. Limitação estrutural do dataset, não um bug.

## Questão 3 – K-Means e Escolha do Melhor K — ✅ Resolvido

- Seleção de atributos com justificativa de negócio por variável (`vars_remover`, com comentário inline em cada remoção — revisado caso a caso, incluindo correção de uma inconsistência entre os critérios usados nos grupos `CASH_ADVANCE` e `PURCHASES`).
- KMeans sem normalização: `K=2` fixado manualmente, consistente com o restante do notebook.
- KMeans com normalização: K do treino vem automaticamente do argmax do Silhouette Score por normalização (não há ambiguidade nesse critério); o gráfico de cotovelo (distortion) tem uma marcação manual editável (`manual_elbow_k_by_normalizer`), só como referência visual — não influencia o K usado no treino.
- Tabela cruzada: usa `CREDIT_LIMIT` discretizado pela regra de Sturges (k bins calculado por `ceil(log2(n)+1)`, não mais um corte arbitrário). Ainda é uma comparação cluster × atributo discretizado do próprio dataset (não uma variável categórica externa real) — mesma limitação estrutural da Q2, documentada e aceita.

## Questão 4 – Comparação K-Means vs DBSCAN — ✅ Implementado + análise abaixo

DBSCAN replica a mesma estrutura do KMeans (scatter + tabela de contingência, sem e com normalização), com metodologia de seleção de `eps`/`min_samples` documentada em markdown (k-distance plot, range por percentil 10-95, cotovelo marcado manualmente, `min_samples` testado em grade, filtro de `noise_pct < 30%` antes de escolher por Silhouette).

**Análise comparativa (Q4):**
- **Quantidade de clusters**: KMeans fixa um K (decidido manualmente ou pelo Silhouette); DBSCAN varia de 2 a mais de 10 dependendo de `eps`/`min_samples`, mas as combinações de menor ruído convergem tipicamente para 2-3 clusters, próximo da escolha do KMeans.
- **Ruído**: diferença estrutural principal — KMeans sempre atribui todo ponto a um cluster; DBSCAN classifica uma fração variável como ruído (de ~5% a >90% nas combinações testadas). Combinações com Silhouette muito alto e ruído extremo (>90%) são descartadas por serem clusters triviais de poucos pontos muito compactos, não um agrupamento útil.
- **Formato**: KMeans assume clusters convexos (baseados em centróide); DBSCAN pode capturar formatos arbitrários por densidade, mas neste dataset os melhores resultados (baixo ruído) tendem a formar grupos compactos similares aos do KMeans — não há evidência de estrutura não-convexa relevante.
- **Sensibilidade à escala**: ambos sensíveis — confirmado pela necessidade de normalizar pra obter `eps` e K coerentes em ambos os algoritmos (sem normalização, distâncias dominadas por `BALANCE`/`PURCHASES`).
- **Facilidade de interpretação**: KMeans é mais simples (K fixo, centróides interpretáveis); DBSCAN exige tunar 2 hiperparâmetros sensíveis e lidar com o conceito extra de ruído.
- **Conclusão**: para este dataset, o **KMeans é mais coerente e estável** — não depende de hiperparâmetros tão sensíveis quanto `eps`, e sempre produz uma partição completa. O DBSCAN só atinge desempenho comparável quando uma fração grande dos pontos é tratada como ruído, sugerindo que a base não tem "vales de baixa densidade" bem definidos que favoreçam o DBSCAN.

## Questão 5 – Impacto da Normalização — ✅ Implementado + análise abaixo

Z-score, Min-Max, Log1p (+ RobustScaler) testados para KMeans, DBSCAN e Hierárquica, com seções "sem"/"com" normalização lado a lado.

**Análise comparativa (Q5):**
- **Gráficos de dispersão**: mudam visivelmente — sem normalização, os clusters do KMeans/DBSCAN se distribuem ao longo dos eixos das variáveis de maior escala; normalizado, a separação fica mais equilibrada entre todas as features.
- **Melhor K/eps**: sem normalização, o DBSCAN não encontrava cluster nenhum com o range de `eps` original (problema já corrigido); com normalização, cada técnica produziu um `eps` ótimo de escala completamente diferente (ex.: MinMaxScaler em torno de 0.01-0.2, RobustScaler/Log1p em 0.5-2, refletindo a escala própria de cada transformação).
- **Silhouette Score**: variou consideravelmente entre normalizações no DBSCAN (RobustScaler e Log1p tendem a Silhouette mais alto com ruído mais baixo do que StandardScaler/MinMaxScaler nas combinações testadas).
- **Distribuição dos clusters / tabela cruzada**: muda de formato junto com o K escolhido por normalização.
- **Conclusão**: a normalização **foi essencial pro DBSCAN** (sem ela, nenhuma combinação de `eps` testada originalmente formava cluster) e teve **impacto moderado no KMeans** (o K sugerido pelo Silhouette varia por normalização, mas K=2 aparece como razoável na maioria dos casos, incluindo o baseline sem normalização).

## Questão 6 – Hierárquica e Dendrograma — ✅ Resolvido

- `compare_linkages` simplificado para variar apenas o método de ligação (`ward`, `complete`, `average`, `single`) com métrica `euclidean` fixa, alinhado literalmente ao texto do PDF (antes testava `manhattan`/`cosine` também, o que não foi pedido).
- **Normalização aplicada**: `X_cut_scaled` (RobustScaler) usado tanto em `compare_linkages` quanto no `linkage()` dos dendrogramas — resolve o gap antigo (rodava em dados brutos).
- **Corte do dendrograma**: implementado e documentado em markdown — maior salto entre alturas de fusão consecutivas nas últimas 20 fusões, indicando o número de clusters sugerido por método de ligação. É automático (sem ambiguidade de hiperparâmetro, diferente do cotovelo do DBSCAN).
- **Comparação com K do KMeans**: os métodos `ward` e `complete` consistentemente sugerem cortes compatíveis com K=2, o mesmo valor usado no KMeans — a clusterização hierárquica **confirma** o resultado do KMeans para este dataset.

## Questão 7 – PCA + Classificação Supervisionada — ✅ Implementado, com ressalva crítica

- Target gerado por modelo configurável (`cluster_target_models`/`TARGET_MODEL`, padrão `agglomerative_ward` com `n_clusters=2`), normalizado com RobustScaler (`X_cut_scaled`) antes de clusterizar — corrige a distorção de escala que inflava artificialmente os resultados na primeira versão.
- Split supervisionado dedicado, estratificado pela variável alvo.
- KNN: 3 normalizações pedidas (Z-score, Min-Max, Log) × 2 representações (todas as colunas / top 5 PCA), com `classification_report` completo por classe (não só agregado) + matriz de confusão pra cada uma das 6 combinações.
- Árvore de Decisão: treinada com e sem as 5 componentes do PCA, com `Pipeline` própria (`VarianceThreshold` + árvore) e visualização gráfica da árvore (top 3 níveis) para as duas variantes.
- Tabela final comparando a melhor configuração do KNN com a melhor da Árvore, e conclusão automática (a partir dos números, não hardcoded) sobre se o PCA ajudou ou não.

**Ressalva crítica**: mesmo após normalizar a geração do target, as métricas de classificação tendem a ficar altas (90%+) — isso é **esperado e estrutural**, não um erro: o desenho da questão pede para classificar o próprio resultado de uma clusterização usando features semelhantes às que geraram esse resultado, então a tarefa é inerentemente mais fácil do que um problema de classificação com rótulos verdadeiramente independentes das features. Vale mencionar isso na apresentação como achado crítico.

## Critérios de Avaliação — status atualizado

| Critério | Pontos | Status |
|---|---|---|
| Análise exploratória | 0,5 | ✅ Completo |
| Análise visual com 2 atributos | 0,5 | ⚠️ Falta coloração por variável de referência (limitação do dataset, não solucionável) |
| K-Means + cotovelo + Silhouette + crosstab | 2,0 | ✅ Resolvido — K manual/automático bem documentado, seleção de variáveis justificada |
| Comparação K-Means vs DBSCAN | 2,0 | ✅ Implementado + análise escrita nesta seção |
| Impacto da normalização | 1,0 | ✅ Implementado + análise escrita nesta seção |
| Hierárquica e dendrograma | 1,0 | ✅ Resolvido — normalização, corte automático e comparação com KMeans, todos presentes |
| PCA + classificação supervisionada (Q7) | 2,0 | ✅ Completo, com ressalva crítica documentada |
| Organização/qualidade do código | 1,0 | ✅ Bom — refatoração feita (helpers reutilizáveis, sem duplicação), hiperparâmetros manuais documentados e editáveis, justificativa de negócio por variável removida |

## Pontos remanescentes (não são gaps de implementação)

- Ausência de variável categórica real no dataset afeta Q2.4 e a "tabela cruzada com variável de referência" das Q3-Q5 — estrutural, sem solução dentro do dataset disponível.
- Métricas altas na Q7 são esperadas dado o desenho da questão — documentado como achado crítico, não como erro a corrigir.
