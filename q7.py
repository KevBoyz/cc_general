import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from q1 import load_data, preprocess_data
from q6 import get_target_labels


def plot_top_pca_components(X, pca, save_path, n_components=None, top_n=10):
    X_scaled = RobustScaler().fit_transform(X)

    if not hasattr(pca, "components_"):
        pca.n_components = n_components
        pca.fit(X_scaled)

    explained_var = pca.explained_variance_ratio_

    result_df = pd.DataFrame({
        "Componente": [f"PC{i}" for i in range(1, len(explained_var) + 1)],
        "Variancia_Explicada": explained_var,
        "Variancia_Acumulada": np.cumsum(explained_var)
    })
    result_df = result_df.sort_values("Variancia_Explicada", ascending=False).head(top_n)

    fig = plt.figure(figsize=(10, 5))
    plt.bar(result_df["Componente"], result_df["Variancia_Explicada"])
    plt.xlabel("Componente Principal")
    plt.ylabel("Variância Explicada")
    plt.title(f"Top {top_n} Componentes Mais Importantes")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    return result_df


def plot_pca_3d(X, pca, save_path, y=None, scaler=None):
    scaler = scaler or RobustScaler()
    X_scaled = scaler.fit_transform(X)

    if not hasattr(pca, "components_"):
        pca.n_components = 3
        X_pca = pca.fit_transform(X_scaled)
    else:
        X_pca = pca.transform(X_scaled)[:, :3]

    var = pca.explained_variance_ratio_[:3]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if y is not None:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, alpha=0.7, cmap="tab10")
        plt.colorbar(scatter)
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.7)

    ax.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.set_zlabel(f"PC3 ({var[2]:.1%})")
    ax.set_title(f"PCA 3D - Variância acumulada: {var[:3].sum():.1%}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    return X_pca, pca


def plot_pca_2d(X, pca, save_path, y=None, scaler=None):
    scaler = scaler or RobustScaler()
    X_scaled = scaler.fit_transform(X)

    if not hasattr(pca, "components_"):
        pca.n_components = 2
        X_pca = pca.fit_transform(X_scaled)
    else:
        X_pca = pca.transform(X_scaled)[:, :2]

    var = pca.explained_variance_ratio_[:2]

    fig, ax = plt.subplots(figsize=(10, 7))

    if y is not None:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.7, cmap="tab10")
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)

    ax.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.set_title(f"PCA 2D - Variância acumulada: {var[:2].sum():.1%}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    return X_pca


def run_knn_classification(X_train, X_test, y_train, y_test):
    knn_normalizers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "Log1p": None
    }

    knn_results = []
    knn_confusion = {}

    for norm_name, normalizer in knn_normalizers.items():
        if normalizer is None:
            X_train_proc = np.log1p(X_train)
            X_test_proc = np.log1p(X_test)
        else:
            X_train_proc = normalizer.fit_transform(X_train)
            X_test_proc = normalizer.transform(X_test)

        pca5 = PCA(n_components=5, random_state=77)
        X_train_pca = pca5.fit_transform(X_train_proc)
        X_test_pca = pca5.transform(X_test_proc)

        feature_sets = {
            "Todas as colunas": (X_train_proc, X_test_proc),
            "Top 5 PCA": (X_train_pca, X_test_pca),
        }

        for feature_set_name, (X_tr, X_te) in feature_sets.items():
            knn = KNeighborsClassifier()
            knn.fit(X_tr, y_train)
            y_pred = knn.predict(X_te)

            print(f"=== KNN | {norm_name} | {feature_set_name} ===")
            print(classification_report(y_test, y_pred))

            report = classification_report(y_test, y_pred, output_dict=True)
            knn_confusion[(norm_name, feature_set_name)] = confusion_matrix(y_test, y_pred)

            knn_results.append({
                "normalizacao": norm_name,
                "features": feature_set_name,
                "accuracy": report["accuracy"],
                "precision": report["weighted avg"]["precision"],
                "recall": report["weighted avg"]["recall"],
                "f1": report["weighted avg"]["f1-score"],
            })

    knn_results_df = pd.DataFrame(knn_results).sort_values("f1", ascending=False).reset_index(drop=True)
    return knn_results_df, knn_confusion


def plot_knn_confusion_matrices(knn_confusion, save_path):
    feature_set_names = ["Todas as colunas", "Top 5 PCA"]
    norm_names = ["StandardScaler", "MinMaxScaler", "Log1p"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for row, feature_set_name in enumerate(feature_set_names):
        for col, norm_name in enumerate(norm_names):
            ax = axes[row, col]
            cm = knn_confusion[(norm_name, feature_set_name)]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
            ax.set_title(f"{norm_name}\n{feature_set_name}")
            ax.set_xlabel("Predito")
            ax.set_ylabel("Real")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def run_tree_classification(X_train_clf, X_test_clf, y_train_clf, y_test_clf, save_dir):
    scaler_for_tree_pca = StandardScaler()
    X_train_tree_scaled = scaler_for_tree_pca.fit_transform(X_train_clf)
    X_test_tree_scaled = scaler_for_tree_pca.transform(X_test_clf)

    pca5_tree = PCA(n_components=5, random_state=77)
    X_train_tree_pca = pca5_tree.fit_transform(X_train_tree_scaled)
    X_test_tree_pca = pca5_tree.transform(X_test_tree_scaled)

    tree_feature_sets = {
        "Todas as colunas": (X_train_clf, X_test_clf),
        "Top 5 PCA": (X_train_tree_pca, X_test_tree_pca),
    }

    tree_results = []
    tree_models = {}

    file_suffix = {"Todas as colunas": "todas_colunas", "Top 5 PCA": "top5_pca"}

    for feature_set_name, (X_tr, X_te) in tree_feature_sets.items():
        tree_pipeline = Pipeline([
            ("feature_selection", VarianceThreshold(threshold=0.01)),
            ("tree", DecisionTreeClassifier(random_state=77)),
        ])

        tree_pipeline.fit(X_tr, y_train_clf)
        y_pred_tree = tree_pipeline.predict(X_te)

        print(f"=== Árvore de Decisão | {feature_set_name} ===")
        print(classification_report(y_test_clf, y_pred_tree))

        report = classification_report(y_test_clf, y_pred_tree, output_dict=True)
        tree_cm = confusion_matrix(y_test_clf, y_pred_tree)
        tree_models[feature_set_name] = tree_pipeline

        tree_results.append({
            "features": feature_set_name,
            "accuracy": report["accuracy"],
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"],
        })

        fig = plt.figure(figsize=(5, 4))
        sns.heatmap(tree_cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.title(f"Árvore de Decisão - Matriz de Confusão ({feature_set_name})")
        plt.savefig(f"{save_dir}/matriz_confusao_arvore_{file_suffix[feature_set_name]}.png")
        plt.close(fig)

    tree_results_df = pd.DataFrame(tree_results).sort_values("f1", ascending=False).reset_index(drop=True)
    return tree_results_df, tree_models


def plot_decision_trees(tree_models, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    for ax, feature_set_name in zip(axes, tree_models.keys()):
        tree_model = tree_models[feature_set_name].named_steps["tree"]
        plot_tree(tree_model, max_depth=3, filled=True, fontsize=8, ax=ax)
        ax.set_title(f"Árvore - {feature_set_name} (top 3 níveis)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def print_final_comparison(knn_results_df, tree_results_df):
    best_knn = knn_results_df.iloc[0]
    best_tree = tree_results_df.iloc[0]

    final_comparison = pd.DataFrame([
        {
            "modelo": f"KNN ({best_knn['normalizacao']}, {best_knn['features']})",
            "accuracy": best_knn["accuracy"],
            "precision": best_knn["precision"],
            "recall": best_knn["recall"],
            "f1": best_knn["f1"],
        },
        {
            "modelo": f"Árvore de Decisão ({best_tree['features']})",
            "accuracy": best_tree["accuracy"],
            "precision": best_tree["precision"],
            "recall": best_tree["recall"],
            "f1": best_tree["f1"],
        },
    ])

    f1_pca_knn = knn_results_df[knn_results_df["features"] == "Top 5 PCA"]["f1"].mean()
    f1_full_knn = knn_results_df[knn_results_df["features"] == "Todas as colunas"]["f1"].mean()

    f1_pca_tree = tree_results_df[tree_results_df["features"] == "Top 5 PCA"]["f1"].mean()
    f1_full_tree = tree_results_df[tree_results_df["features"] == "Todas as colunas"]["f1"].mean()

    print(f"KNN    - F1 médio (todas as colunas): {f1_full_knn:.4f} | F1 médio (top 5 PCA): {f1_pca_knn:.4f}")
    print(f"Árvore - F1 (todas as colunas): {f1_full_tree:.4f} | F1 (top 5 PCA): {f1_pca_tree:.4f}")
    print(
        "PCA manteve/melhorou o desempenho médio do KNN em relação a usar todas as colunas."
        if f1_pca_knn >= f1_full_knn else
        "PCA reduziu o desempenho médio do KNN em relação a usar todas as colunas."
    )
    print(
        "PCA manteve/melhorou o desempenho da Árvore em relação a usar todas as colunas."
        if f1_pca_tree >= f1_full_tree else
        "PCA reduziu o desempenho da Árvore em relação a usar todas as colunas."
    )

    print(final_comparison)


def pipeline():
    os.makedirs("image/q7", exist_ok=True)

    df = preprocess_data(load_data())
    target_labels, X_cut_features = get_target_labels(df)

    shared_pca = PCA(random_state=77)

    result_df = plot_top_pca_components(
        X_cut_features, pca=shared_pca, top_n=10, save_path="image/q7/top_pca_componentes.png"
    )
    print(result_df)

    X_pca_3d, pca_3d = plot_pca_3d(
        X_cut_features, pca=shared_pca, y=target_labels, save_path="image/q7/pca_3d.png"
    )
    print((X_pca_3d, pca_3d))

    X_pca_2d = plot_pca_2d(
        X_cut_features, pca=shared_pca, y=target_labels, save_path="image/q7/pca_2d.png"
    )
    print(X_pca_2d)

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_cut_features, target_labels,
        test_size=0.2, random_state=77, stratify=target_labels
    )

    knn_results_df, knn_confusion = run_knn_classification(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
    print(knn_results_df)

    plot_knn_confusion_matrices(knn_confusion, save_path="image/q7/matrizes_confusao_knn.png")

    tree_results_df, tree_models = run_tree_classification(
        X_train_clf, X_test_clf, y_train_clf, y_test_clf, save_dir="image/q7"
    )
    print(tree_results_df)

    plot_decision_trees(tree_models, save_path="image/q7/arvores_decisao.png")

    print_final_comparison(knn_results_df, tree_results_df)


if __name__ == "__main__":
    pipeline()
