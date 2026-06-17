# Questão 7 — PCA + Classificação Supervisionada (KNN e Árvore de Decisão)

## Contexto

A seção `## PCA E classificação supervisionada` do notebook já tem PCA exploratório (`plot_pca_3d`, `plot_top_pca_components`) e uma nota antiga (`71c6dc03`) dizendo a intenção: gerar uma coluna `cluster` a partir do "melhor modelo de clusterização encontrado" e usá-la como target para KNN e Árvore de Decisão. Isso nunca foi implementado — é exatamente a Questão 7 do PDF, pendente desde o relatório de conformidade.

O PDF pede: separar entrada/target, aplicar pré-processamento (Z-score, Min-Max, Log se fizer sentido), aplicar PCA, treinar um classificador com dados originais e com dados reduzidos via PCA, comparar com acurácia/precisão/recall/F1/matriz de confusão.

**Decisão confirmada com o usuário**: como `AgglomerativeClustering` não tem `predict()` para dados novos, vou clusterizar `X_train_cut + X_test_cut` juntos (Ward) para gerar o rótulo de cada linha, e só depois fazer um `train_test_split` novo (features + rótulo) específico para a etapa supervisionada — sem reaproveitar o split usado nas seções não-supervisionadas.

## O que será implementado

Tudo dentro da seção `## PCA E classificação supervisionada`, depois das células de PCA exploratório já existentes.

### 1. Geração do target (configurável)

Um dicionário `cluster_target_models` mapeando nome → estimador (ex.: `"agglomerative_ward": AgglomerativeClustering(n_clusters=2, linkage="ward")`, e mais entradas tipo `"kmeans"`, `"dbscan"` prontas para troca futura), e uma variável única `TARGET_MODEL = "agglomerative_ward"` no topo. Trocar o modelo que gera o target = mudar essa string.

```python
X_all = pd.concat([X_train_cut.drop(columns=["cluster"]), X_test_cut.drop(columns=["cluster"])])
target_labels = cluster_target_models[TARGET_MODEL].fit_predict(X_all)
```

`n_clusters=2` porque foi o valor sugerido tanto pelo corte do dendrograma Ward (seção anterior) quanto pelo KMeans baseline — mantém os resultados comparáveis com o resto do notebook.

### 2. Split supervisionado novo

`train_test_split(X_all, target_labels, test_size=0.2, random_state=77, stratify=target_labels)` — split dedicado à tarefa supervisionada, estratificado para não distorcer proporções de cluster.

### 3. KNN — 3 normalizações × 2 representações de features

Reaproveita o dict `normalizers` já existente (`StandardScaler`, `RobustScaler` → não pedido pelo PDF mas já há no notebook; vou usar só os 3 pedidos: `StandardScaler` (Z-score), `MinMaxScaler` (Min-Max), `Log1p`), e o `PCA` já importado.

Para cada normalização:
- Ajusta o normalizador (e PCA com `n_components=5`) **só no treino**, transforma treino e teste (evita leakage).
- Treina `KNeighborsClassifier()` (default `n_neighbors=5`) com (a) todas as colunas normalizadas e (b) só os 5 componentes do PCA.
- Calcula accuracy, precision, recall, f1 (`average="weighted"`) e matriz de confusão.

Resultado: tabela única com as 6 combinações (3 normalizações × 2 representações) ordenada por F1, e uma grade 2×3 de matrizes de confusão (linhas = todas-colunas / PCA-5, colunas = normalização) — mesmo padrão visual de grade já usado em outras seções do notebook (`plot_scatter_by_group` etc., mas aqui inline já que o conteúdo é diferente o suficiente).

### 4. Árvore de Decisão com Pipeline própria

Árvore não precisa de normalização (conforme a própria nota do time), mas o pedido é uma `Pipeline` com seleção de features + pré-processamento dedicados:

```python
tree_pipeline = Pipeline([
    ("feature_selection", VarianceThreshold(threshold=0.01)),
    ("tree", DecisionTreeClassifier(random_state=77)),
])
```

`VarianceThreshold` reaproveita o achado já registrado no notebook (`32131d1d`) de que `TENURE` tem variância baixíssima — aqui isso é tratado automaticamente em vez de manualmente. Treina no split supervisionado, mesmas métricas (accuracy/precision/recall/F1/confusion matrix) para comparar com o melhor resultado do KNN.

### 5. Comparação final

Tabela final juntando a melhor configuração do KNN + a Árvore, respondendo à pergunta do PDF: "vale a pena reduzir dimensionalidade neste caso?".

## Arquivo afetado

- `analysis.ipynb` — novas células inseridas na seção `## PCA E classificação supervisionada`, depois de `36788073` (última célula atual da seção). Nenhuma célula existente é alterada.

## Verificação

- Rodar o notebook (ou um script equivalente via `.venv`) do ponto da Questão 7 em diante e confirmar que todas as células executam sem erro.
- Conferir que a tabela de comparação do KNN tem 6 linhas com métricas plausíveis (entre 0 e 1) e que a Árvore de Decisão produz métricas comparáveis.
- Confirmar que trocar `TARGET_MODEL` para outra chave do dicionário (ex.: `"kmeans"`) faz a célula de geração de target rodar sem precisar editar mais nada abaixo.
