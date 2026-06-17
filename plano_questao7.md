# Plano: remoção do split inicial + melhorias de EDA/gráficos + Questão 7 (PCA + Classificação)

## Contexto

Discussão prévia concluiu duas coisas:

1. **O PDF só exige divisão treino/teste na Questão 7** (avaliação de generalização do classificador supervisionado). Nada em Q1-Q6 pede isso, e o split atual (`X_train`/`X_test`, feito logo após a limpeza) só estava sendo útil pra ajustar o imputer sem leakage — o resto das seções não-supervisionadas nunca avaliou `X_test_cut` separadamente de forma significativa. Decisão: **remover o split inicial**, fazer Q1-Q6 inteiramente sobre o dataset completo, e introduzir **um único split, no momento da Q7**, especificamente para avaliar KNN/Árvore.
2. Isso também simplifica a geração do target da Q7: como não há mais split prévio, não precisamos mais "juntar treino+teste de volta" — é só `fit_predict` direto no dataset completo.

Além disso, o usuário pediu uma lista de melhorias em gráficos/tabelas já existentes (discretização, marcação de cotovelo do DBSCAN, dendrogramas sem grid, histogramas completos, boxplot com escalas individuais, variância acumulada do PCA) e que o PCA 3D já existente passe a colorir pelos clusters do Ward.

Este documento é só o **plano** — nada será implementado até confirmação.

## 1. Remoção do split inicial (afeta quase todo o notebook antes da Q7)

**Renomeação consistente** (find-and-replace cuidadoso célula por célula):
- `X_train`/`X_test` (criados por `train_test_split` em `9f7d30d8`) → deixam de existir; tudo que os usava passa a usar `df` diretamente.
- `X_train_cut`/`X_test_cut` (pós seleção de variáveis, `764a17bc`) → renomeados para um único `X_cut`.
- `X_train_cut_scaled` (seção hierárquica) → `X_cut_scaled`.

**Mudanças de lógica decorrentes:**
- `9f7d30d8`: remove o `train_test_split`; a imputação passa a ser `df[columns_with_missing_values] = pipeline.fit_transform(df[columns_with_missing_values])` (fit direto em `df`, sem split).
- `764a17bc` (`remove_correlated_features`): perde o parâmetro/retorno `X_test`; só recebe e retorna `X_cut`.
- `4a88a9ca` (KMeans baseline): só uma linha de predict (`X_cut["cluster"] = model.predict(X_cut)`), sem a linha gêmea de teste.
- `0cfd8f39`, `2fc71123`, `6df582a1`, `9776a5f6`, `1e2914fa`, `e372f23f`, `ce23b785`, `5626edf5`, `c0200a86`, `0b716cbe`: todas trocam `X_train_cut`/`X_train_cut_scaled` por `X_cut`/`X_cut_scaled`.
- **Bônus, já que essas células serão tocadas de qualquer forma**: corrijo o vazamento da coluna `cluster` (gerada pelo KMeans baseline) que hoje entra como feature nas seções de DBSCAN e hierárquica sem querer — passam a usar `X_cut.drop(columns=["cluster"])` (mesma correção que já tinha sido feita só pro `compare_preprocessing_kmeans`).

**Resultado esperado**: como agora há ~8.949 linhas em vez de ~7.159 (estava usando só os 80% de treino), os números de silhouette/eps/cotovelo de todas as seções (KMeans, DBSCAN, hierárquica) vão mudar um pouco — isso é esperado e correto, não é regressão.

## 2. Melhorias na Análise Exploratória (Q1)

### 2.1 Histogramas — cobrir todas as variáveis contínuas
Célula `68418bae` hoje só tem 6 das 14 variáveis contínuas (`BALANCE`, `PURCHASES`, `CASH_ADVANCE`, `PAYMENTS`, `MINIMUM_PAYMENTS`, `CREDIT_LIMIT`). Vou expandir `continuous_cols` para as 14 (todas as `float64` exceto o que não fizer sentido), grid 4×4 (16 posições, 2 ocultas), mesmo padrão de "esconder eixos extras" já usado no grid de KDE.

### 2.2 Distribuição de variáveis discretas
Célula `0e847256` já cobre as 3 variáveis discretas existentes (`CASH_ADVANCE_TRX`, `PURCHASES_TRX`, `TENURE`) — **já está completo**, nenhuma mudança necessária aqui.

### 2.3 Novo boxplot com escalas individuais
**Sem alterar** a célula `d6634f73` (boxplot único combinado, que fica como está). Cria uma **célula nova logo abaixo**: grid de subplots (um por coluna de `df`, escala própria de cada eixo y), tamanho 4×5 (17 colunas, 3 ocultas), seguindo o mesmo padrão de grid das outras seções.

## 3. Discretização por regra de Sturges (tabelas de contingência)

Célula `34fe8bc8` hoje usa bins fixos arbitrários (`[0, 0.2, 0.4, 0.6, 0.8, 1.0]`, 5 categorias) para discretizar `PRC_FULL_PAYMENT`. Troca para a regra de Sturges:

```python
n = len(X_cut)
k_bins = int(np.ceil(np.log2(n) + 1))
h = (X_cut["PRC_FULL_PAYMENT"].max() - X_cut["PRC_FULL_PAYMENT"].min()) / k_bins
bins = np.linspace(X_cut["PRC_FULL_PAYMENT"].min(), X_cut["PRC_FULL_PAYMENT"].max(), k_bins + 1)
```
Com ~8.949 linhas, `k_bins = ceil(log2(8949)+1) ≈ 15`. Isso troca as tabelas de contingência de 5 para ~15 categorias — mais granular e estatisticamente justificado, mas as heatmaps de contingência ficam mais largas (vou ajustar `figsize` onde necessário pra não ficar ilegível). Essa célula alimenta todas as tabelas de contingência (KMeans e DBSCAN, sem e com normalização), então a mudança se propaga automaticamente.

## 4. Marcar o cotovelo nos gráficos de k-distância do DBSCAN

Hoje (`48d4715d` sem normalização, `9776a5f6` com normalização) os gráficos de k-distância não destacam nenhum ponto. Vou usar `KneeLocator` (pacote `kneed`, já instalado como dependência do Yellowbrick) pra achar o ponto de maior curvatura:

```python
from kneed import KneeLocator
kl = KneeLocator(range(len(k_distances)), k_distances, curve="convex", direction="increasing")
elbow_idx, elbow_value = kl.elbow, k_distances[kl.elbow]
```
Marca esse ponto com um `scatter`/`axhline` destacado e anota o valor sugerido de `eps` no gráfico — nas duas versões (sem normalização: 1 gráfico; com normalização: 4 painéis, um cotovelo por normalização).

## 5. Dendrogramas sem grid

Célula `5626edf5`: adiciona `plt.grid(False)` explícito depois do `dendrogram(...)`, garantindo que nenhuma gridline apareça independente de estilo herdado.

## 6. PCA: tabela de variância acumulada

Na seção de PCA (`0b716cbe`/`36788073`), além do gráfico de barras já existente com a variância de cada componente, adiciona uma tabela (`DataFrame`) com `Variancia_Explicada` e `Variancia_Acumulada` (`np.cumsum`) para os top 10 componentes — ajuda a justificar visualmente por que 5 componentes foram escolhidos pra Q7.

## 7. PCA 3D coloria pelos clusters do Ward

A célula `c0200a86` (chamada de `plot_pca_3d`) hoje colore pelos clusters do KMeans baseline (`X_cut["cluster"]`). Vou mover a geração do target da Q7 (ver seção 8.1) para **antes** dessa célula, e trocar a chamada para usar esses novos rótulos do Ward em vez do KMeans baseline — assim a visualização 3D já reflete o modelo de clusterização que vai servir de target supervisionado.

## 8. Questão 7 — PCA + Classificação Supervisionada

### 8.1 Geração do target (configurável), agora simplificada
Sem split prévio, não tem mais "juntar treino+teste":
```python
cluster_target_models = {
    "agglomerative_ward": AgglomerativeClustering(n_clusters=2, linkage="ward"),
    "kmeans": make_kmeans(n_clusters=2),
    # fácil adicionar outras entradas aqui
}
TARGET_MODEL = "agglomerative_ward"
target_labels = cluster_target_models[TARGET_MODEL].fit_predict(X_cut.drop(columns=["cluster"]))
```
Essa célula passa a ficar logo depois da markdown `71c6dc03`, antes do `plot_pca_3d` (item 7).

### 8.2 Split supervisionado — o único split do notebook
```python
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_cut.drop(columns=["cluster"]), target_labels,
    test_size=0.2, random_state=77, stratify=target_labels
)
```

### 8.3 KNN — 3 normalizações × 2 representações
Para `StandardScaler` (Z-score), `MinMaxScaler` (Min-Max) e `Log1p`: ajusta normalizador (e `PCA(n_components=5)`) só no treino, transforma treino/teste, treina `KNeighborsClassifier()` com (a) todas as colunas normalizadas e (b) só os 5 componentes do PCA. Calcula **`classification_report`** completo (por classe + agregado, não só um número médio) + matriz de confusão, para as 6 combinações. Tabela resumo ordenada por F1 + grade 2×3 de matrizes de confusão.

### 8.4 Árvore de Decisão com Pipeline própria
```python
tree_pipeline = Pipeline([
    ("feature_selection", VarianceThreshold(threshold=0.01)),
    ("tree", DecisionTreeClassifier(random_state=77)),
])
```
`VarianceThreshold` reaproveita o achado já registrado (`32131d1d`) de que `TENURE` tem variância baixíssima. Mesmas métricas completas (`classification_report` + confusion matrix) no mesmo split de 8.2.

### 8.5 Comparação final
Tabela juntando a melhor configuração do KNN + a Árvore, respondendo: vale a pena reduzir dimensionalidade aqui?

## Arquivo afetado

- `analysis.ipynb` — ~25 células editadas (renomeação de variáveis + correção do vazamento de `cluster`) e ~10 células novas inseridas (histogramas completos, boxplot novo, cotovelo do DBSCAN, variância acumulada, geração de target, split supervisionado, KNN, Árvore, comparação final).

## Verificação

- Rodar o notebook (ou script equivalente via `.venv`) do início ao fim sem erros.
- Confirmar que nenhuma célula ainda referencia `X_train`/`X_test`/`X_train_cut`/`X_test_cut` (grep no notebook).
- Conferir que a tabela de contingência com Sturges não ficou ilegível (ajustar `figsize` se necessário).
- Conferir que o cotovelo marcado nos gráficos de k-distância corresponde a um valor de eps plausível (dentro da faixa já usada na busca por percentil).
- Conferir que a tabela final do KNN tem 6 linhas com métricas entre 0 e 1, e que trocar `TARGET_MODEL` continua funcionando sem editar mais nada.
