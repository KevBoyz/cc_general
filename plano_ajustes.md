# Plano de Ajustes — revisão pós-implementação (Questões 1-7)

Cobre todos os itens levantados em `coisas.txt` + decisões adicionais desta rodada. Nada implementado ainda.

## 0. Verificação já feita (sem necessidade de ação)

Computei os pares de variáveis correlacionadas no dataset completo, em duas rodadas: primeiro com threshold 0.80 (9 pares, função antiga `remove_correlated_features`), depois com threshold 0.75 (14 pares, lista manual `vars_remover` atual). Em ambos os casos, **todos os pares têm pelo menos uma variável removida** — nenhum par escapou, não há necessidade de adicionar mais nada à lista além do ajuste do item 1.

## 1. Seleção de variáveis — já resolvido pelo usuário, com um ajuste

O usuário substituiu `remove_correlated_features` por `show_correlated_pairs` (só lista os pares, threshold agora 0.75) + uma lista manual `vars_remover` com justificativa de negócio inline em cada item — não preciso mais adicionar a markdown de justificativa, já está feita.

Revisei caso a caso (ver conversa) e encontrei uma inconsistência: pra `CASH_ADVANCE` o critério inverteu (manteve frequência, descartou o valor monetário), diferente do grupo `PURCHASES` (manteve valor, descartou frequência/contagem). **Ajuste a ser aplicado**: trocar `CASH_ADVANCE_FREQUENCY` por `CASH_ADVANCE` na lista de removidas — o valor monetário do adiantamento é mais informativo pra risco de crédito do que a frequência normalizada, e fica consistente com o critério já usado no grupo `PURCHASES`.

```python
vars_remover = [
    'CASH_ADVANCE_TRX',                   # contagem bruta → quase idêntica a CASH_ADVANCE_FREQUENCY (corr 0.98)
    'CASH_ADVANCE_FREQUENCY',             # frequência → substituída por CASH_ADVANCE (valor monetário mais informativo pra risco)
    'MINIMUM_PAYMENTS',                   # derivada → substituída por BALANCE
    'PURCHASES_TRX',                      # contagem bruta → substituída por PURCHASES
    'ONEOFF_PURCHASES',                   # subcomponente → capturado por PURCHASES
    'PURCHASES_FREQUENCY',                # frequência → optou-se pelo montante PURCHASES
    'INSTALLMENTS_PURCHASES',             # subcomponente → capturado por PURCHASES
    'PURCHASES_INSTALLMENTS_FREQUENCY',   # freq. parcelamento → removida em cascata
]
```

Verifiquei computacionalmente: com essa troca, os 14 pares com `|corr| >= 0.75` continuam todos cobertos (nenhum par fica com as duas variáveis mantidas). `X_cut` resultante: `BALANCE`, `BALANCE_FREQUENCY`, `PURCHASES`, `CASH_ADVANCE`, `ONEOFF_PURCHASES_FREQUENCY`, `CREDIT_LIMIT`, `PAYMENTS`, `PRC_FULL_PAYMENT`, `TENURE`.

## 2. Geração do target da Q7 — normalizar com RobustScaler (CORRIGE o achado do item de "resultados bons demais")

A causa raiz dos scores inflados do KNN/Árvore (96-98%) é que o Ward clusteriza `X_cut_features` **sem normalização** — a divisão em 2 clusters fica dominada por colunas de escala grande (`BALANCE`, `PURCHASES`), exatamente o problema de escala já corrigido em outras seções. Trocar para usar `X_cut_scaled` (RobustScaler, já calculado na seção hierárquica, reaproveitado) deixa a clusterização mais equilibrada entre todas as features.

```python
target_labels = cluster_target_models[TARGET_MODEL].fit_predict(X_cut_scaled)
```

Isso não elimina toda a circularidade conceitual (o target ainda vem de clusterização sobre as mesmas features usadas pra classificar — isso é inerente ao que a Q7 pede), mas remove a distorção de escala que tornava a tarefa artificialmente fácil. Vou documentar essa ressalva no relatório de conformidade mesmo assim.

## 3. KMeans com normalização — estrela do cotovelo manual, silhouette continua automático

Correção do usuário: a seleção de K por silhouette (painel direito, que define o K usado no treino via `best_k_by_normalizer`) continua automática — é só pegar o argmax no intervalo, sem ambiguidade. A seleção manual se aplica apenas à estrela do gráfico de **cotovelo** (painel esquerdo, curva de distortion): em vez de vir do `elbow.elbow_value_` (auto-detectado pelo Yellowbrick), agora vem de um dict editável `manual_elbow_k_by_normalizer` (default 2 pra todas). Essa estrela é só informativa — não influencia o K usado no treino.

## 4. DBSCAN sem normalização — cotovelo manual

```python
dbscan_elbow_idx_no_norm = 1  # índice no array k_distances; edite aqui
```
Substitui o `KneeLocator`. Remove o `from kneed import KneeLocator`.

## 5. DBSCAN com normalização — cotovelo manual por normalização

```python
dbscan_elbow_idx_by_normalizer = {
    "StandardScaler": 1,
    "RobustScaler": 1,
    "MinMaxScaler": 1,
    "Log1p": 1,
}
```
Usado nos 4 painéis do gráfico de k-distância em vez do `KneeLocator` por painel.

**Decisão sobre dependências**: `kneed` é removido do projeto (única função era a detecção automática que deixa de existir). `yellowbrick` é mantido, só sem a marcação automática (`locate_elbow=False`) — continua útil para desenhar as curvas de distortion/silhouette.

## 6. Markdown explicando metodologia de seleção de parâmetros do DBSCAN (NOVO)

Célula markdown próxima ao código de busca de `eps`/`min_samples` (seções sem e com normalização), explicando: gráfico de k-distância, percentis 10-95 como range de busca, e agora o cotovelo manual como ponto de referência adicional.

## 7. Hierárquica — simplificar `compare_linkages`

Remover a variação de métrica (`manhattan`, `cosine`) — manter só `euclidean`, variando apenas o método de ligação (`ward`, `complete`, `average`, `single`), que é literalmente o que a Q6 do PDF pede. `RobustScaler` é mantido (não será removido das comparações de normalização do KMeans/DBSCAN).

```python
def compare_linkages(X, n_clusters=3):
    results = []
    for linkage in ["ward", "complete", "average", "single"]:
        labels = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric="euclidean").fit_predict(X)
        results.append({"linkage": linkage, "silhouette": silhouette_score(X, labels)})
    return pd.DataFrame(results).sort_values("silhouette", ascending=False)
```

## 8. Markdown explicando o corte automático do dendrograma (NOVO)

Célula markdown próxima ao loop de dendrogramas (`5626edf5`), explicando a lógica do maior salto entre alturas de fusão consecutivas no topo da hierarquia (esse corte continua automático — só o cotovelo do DBSCAN e o K do KMeans passam a ser manuais).

## 9. Questão 7 — tabela do KNN: remover "_weighted" dos nomes

`precision_weighted` → `precision`, `recall_weighted` → `recall`, `f1_weighted` → `f1` (continua sendo a média weighted do `classification_report`, só não aparece mais no nome da coluna).

## 10. Questão 7 — Árvore de Decisão com e sem PCA-5 + gráfico da árvore

Espelhar a estrutura do KNN: treinar `tree_pipeline` em (a) `X_train_clf`/`X_test_clf` (todas as colunas) e (b) as mesmas 5 componentes do PCA já usadas no KNN. Para cada uma: `classification_report` completo + matriz de confusão + `sklearn.tree.plot_tree(modelo, max_depth=3, ...)` (limita só a profundidade *desenhada*, a árvore treinada continua sem restrição).

Resposta à pergunta sobre `X_cut_features`: remoção de correlacionadas ajuda métodos baseados em distância (KMeans/DBSCAN/hierárquica/KNN), mas **árvore de decisão é robusta a colinearidade** — ela escolhe a melhor feature em cada split e ignora a redundante sem prejuízo. Manter `X_cut_features` não piora nem melhora a árvore de forma relevante; é só consistência com o resto do notebook.

## 11. Atualizar `relatorio_conformidade.md`

- Refletir tudo que foi implementado desde a última versão (histogramas completos, boxplot individual, Sturges, cotovelo do DBSCAN, corte do dendrograma, normalização da hierárquica, Q7 completa).
- Q4 (comparação KMeans×DBSCAN) e Q5 (impacto da normalização): tratar como **análise textual no relatório**, não como código faltante — escrever a comparação ali em vez de pedir mais células no notebook.
- Incluir a ressalva crítica sobre a Q7: mesmo após normalizar a geração do target (item 2), o desenho da questão (classificar o próprio resultado de uma clusterização) tende a gerar métricas altas por natureza; isso é esperado e vale documentar como achado crítico, não como erro.
- Nota sobre simplificações feitas: `compare_linkages` agora só varia linkage (não métrica), no que está mais alinhado ao texto do PDF.

## Resumo do que NÃO muda (decisões já tomadas)

- `RobustScaler` continua nas comparações de normalização do KMeans/DBSCAN (não será removido).
- Corte automático do dendrograma continua automático (só cotovelo DBSCAN e K do KMeans passam a manuais).
- `X_cut_features` (sem as colunas redundantes) continua sendo a base usada em todas as seções, incluindo a Árvore.
