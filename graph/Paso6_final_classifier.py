"""
Paso 6: Clasificador final — umbral óptimo + no lineal + fusión de features
=============================================================================

Tres mejoras sobre el diagnóstico de Paso 5, todas en un solo script:

  ① UMBRAL ÓPTIMO PARA F1
     En vez de clasificar con el umbral por defecto (0.5), se barre la curva
     precision-recall y se encuentra el punto que maximiza F1. Esto es
     especialmente importante en La Victoria (11.2% peligroso) donde el
     umbral 0.5 castiga injustamente el F1 aunque el AUC sea alto.

  ② CLASIFICADORES NO LINEALES
     Además de la regresión logística (baseline lineal), se prueban
     Random Forest y un MLP pequeño — pueden capturar interacciones no
     lineales entre dimensiones del embedding que la regresión logística
     no puede.

  ③ FUSIÓN DE FEATURES VISUAL + ESTRUCTURAL
     Se calculan features puramente topológicas del grafo de calles
     (grado, betweenness, closeness, clustering, grado promedio de vecinos)
     y se comparan 3 conjuntos: solo visual (mean+std de Paso 5),
     solo estructural, y fusionado (concatenado). Esta es la comparación
     de 3 columnas que define el resultado central del proyecto.

ESQUEMA DE VALIDACIÓN CRUZADA — CORREGIDO POR LEAKAGE ESPACIAL (Paso 8):
  Los vecindarios son ego-graphs que se solapan — dos nodos adyacentes
  comparten buena parte de los mismos puntos de crimen e imágenes. Con
  K-fold aleatorio, el modelo "ve" en entrenamiento información casi igual
  a la de test, inflando el AUC. Por eso cada combinación se evalúa con
  DOS esquemas:
    - "spatial"  → GroupKFold con 5 bloques geográficos (KMeans sobre lat/lon).
                   Esta es la métrica PRIMARIA, la que se reporta como resultado.
    - "random"   → StratifiedKFold aleatorio, tradicional.
                   Se mantiene solo como referencia de cuánto inflaba el leakage.

Para cada combinación (3 feature sets × 3 clasificadores = 9 combos) se
evalúa con ambos esquemas (18 evaluaciones), reportando AUC, F1 a umbral
0.5, F1 al umbral óptimo, y accuracy.

REQUISITO PREVIO: Paso 4 y Paso 5 ya corridos (usa sus outputs directamente,
no recalcula tensores ni embeddings).

Uso:
    python Paso6_final_classifier.py --distrito all --year 2016
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                              roc_auc_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED             = 42
N_SPLITS         = 5
N_SPATIAL_BLOCKS = 5


# ─── Grafo y features estructurales ───────────────────────────────────────────

def rebuild_graph(nodes_csv: Path, edges_csv: Path):
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        G.add_node(int(row["node_id"]), x=row["lon"], y=row["lat"])
    for _, row in edges_df.iterrows():
        G.add_edge(int(row["u"]), int(row["v"]), length=row.get("length_m"))
    return G, nodes_df


def compute_structural_features(G: nx.Graph, node_ids: list):
    """
    Features puramente topológicas del grafo de calles, SIN ninguna
    información de crimen — sirven para medir cuánto aporta la estructura
    urbana per se, independiente de lo visual.
    """
    print("  Calculando features estructurales (degree, betweenness, "
          "closeness, clustering, avg_neighbor_degree)…")
    degree              = dict(G.degree())
    betweenness         = nx.betweenness_centrality(G, seed=SEED)
    closeness           = nx.closeness_centrality(G)
    clustering          = nx.clustering(G)
    avg_neighbor_degree = nx.average_neighbor_degree(G)

    rows = []
    for n in node_ids:
        rows.append([
            degree.get(n, 0),
            betweenness.get(n, 0.0),
            closeness.get(n, 0.0),
            clustering.get(n, 0.0),
            avg_neighbor_degree.get(n, 0.0),
        ])
    feature_names = ["degree", "betweenness", "closeness",
                      "clustering", "avg_neighbor_degree"]
    return np.array(rows, dtype=np.float32), feature_names


# ─── Bloques espaciales (para GroupKFold, evita leakage entre vecinos) ──────

def assign_spatial_blocks(nodes_csv: Path, node_ids: list, n_blocks: int = N_SPATIAL_BLOCKS):
    """
    KMeans sobre lat/lon de los nodos → bloques geográficos. Usados como
    'groups' de GroupKFold, así nodos vecinos (que comparten puntos de
    crimen/imágenes en sus tensores) siempre caen en el mismo fold.
    """
    nodes_df = pd.read_csv(nodes_csv).set_index("node_id")
    coords = np.array([[nodes_df.loc[n, "lon"], nodes_df.loc[n, "lat"]] for n in node_ids])
    km = KMeans(n_clusters=n_blocks, random_state=SEED, n_init=10).fit(coords)
    return km.labels_


# ─── Umbral óptimo para F1 ─────────────────────────────────────────────────────

def find_optimal_threshold_f1(y_true: np.ndarray, y_proba: np.ndarray):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    if len(thresholds) == 0:
        return 0.5, 0.0
    best_idx = int(np.argmax(f1s[:-1]))   # último punto no tiene threshold asociado
    return float(thresholds[best_idx]), float(f1s[best_idx])


# ─── Evaluación de una combinación feature_set × clasificador × esquema CV ──

def evaluate_combo(X: np.ndarray, y: np.ndarray, clf, name: str,
                    cv, groups=None, cv_scheme: str = "spatial"):
    """
    cross_val_predict con probabilidades out-of-fold: cada nodo es predicho
    por un modelo que NUNCA lo vio en entrenamiento. `cv` puede ser
    GroupKFold (esquema espacial, PRIMARIO) o StratifiedKFold (aleatorio,
    referencia). Devuelve también y_proba completo (para Paso 7).
    """
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    y_proba = cross_val_predict(pipe, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]

    y_pred_default = (y_proba >= 0.5).astype(int)
    f1_default      = f1_score(y, y_pred_default)
    acc_default     = accuracy_score(y, y_pred_default)
    auc             = roc_auc_score(y, y_proba)

    opt_threshold, f1_optimal = find_optimal_threshold_f1(y, y_proba)
    y_pred_optimal  = (y_proba >= opt_threshold).astype(int)
    acc_optimal     = accuracy_score(y, y_pred_optimal)

    result = {
        "estrategia":        name,
        "cv_scheme":         cv_scheme,
        "n_features":        X.shape[1],
        "auc":                auc,
        "accuracy_default":  acc_default,
        "f1_default":        f1_default,
        "optimal_threshold": opt_threshold,
        "f1_optimal":        f1_optimal,
        "accuracy_optimal":  acc_optimal,
    }
    return result, y_proba


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_pipeline(
    distrito: str,
    year,
    eigen_dir: Path,
    tensors_dir: Path,
    graph_dir: Path,
    output_dir: Path,
    strat_tag: str = "dualgmm",
):
    year_str = str(year) if year else "all"
    base     = f"{distrito.replace(' ', '_')}_{year_str}"
    print(f"\n{'='*60}")
    print(f"  Distrito: {distrito} | Año: {year_str}")
    print(f"{'='*60}")

    # ── Features visuales (ganadoras de Paso 5) y labels ───────────────────
    visual = np.load(eigen_dir / f"{base}_features_mean_std.npy")
    labels = np.load(eigen_dir / f"{base}_labels.npy")
    with open(tensors_dir / f"{base}_node_ids_order.json") as f:
        node_ids = json.load(f)
    print(f"  Visual features: {visual.shape}  |  "
          f"Labels: {int(labels.sum())} peligrosos / {len(labels)}")

    # ── Features estructurales del grafo ────────────────────────────────────
    G, _ = rebuild_graph(
        graph_dir / f"{base}_{strat_tag}_nodes.csv",
        graph_dir / f"{base}_edges.csv",
    )
    structural, structural_names = compute_structural_features(G, node_ids)
    print(f"  Structural features: {structural.shape}  ({', '.join(structural_names)})")

    fused = np.hstack([visual, structural])

    feature_sets = {
        "visual_only":     visual,
        "structural_only": structural,
        "fused":           fused,
    }

    classifiers = {
        "logistic": LogisticRegression(
            C=0.1, max_iter=2000, class_weight="balanced", random_state=SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=SEED, n_jobs=-1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=800,
            early_stopping=True, random_state=SEED,
        ),
    }

    # ── Esquemas de CV: espacial (primario, evita leakage) + aleatorio (referencia) ──
    nodes_csv      = graph_dir / f"{base}_{strat_tag}_nodes.csv"
    spatial_groups = assign_spatial_blocks(nodes_csv, node_ids)
    block_sizes    = pd.Series(spatial_groups).value_counts().sort_index().tolist()
    print(f"  Bloques espaciales (GroupKFold): tamaños {block_sizes}")

    cv_spatial = GroupKFold(n_splits=N_SPATIAL_BLOCKS)
    cv_random  = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    results           = []
    predictions_store = {
        "node_ids": node_ids, "y_true": labels.tolist(),
        "predictions": {}, "thresholds": {},
        "structural_feature_names": structural_names,
    }

    for fs_name, X in feature_sets.items():
        for clf_name, clf in classifiers.items():
            combo_name = f"{fs_name} + {clf_name}"

            # Esquema PRIMARIO: bloques espaciales (sin leakage)
            print(f"  Evaluando: {combo_name}  [spatial]…")
            result_sp, y_proba_sp = evaluate_combo(
                X, labels, clf, combo_name, cv=cv_spatial, groups=spatial_groups,
                cv_scheme="spatial",
            )
            results.append(result_sp)
            key_sp = f"{combo_name} | spatial"
            predictions_store["predictions"][key_sp] = y_proba_sp.tolist()
            predictions_store["thresholds"][key_sp]  = result_sp["optimal_threshold"]

            # Esquema de REFERENCIA: K-fold aleatorio (para medir el gap de leakage)
            print(f"  Evaluando: {combo_name}  [random]…")
            result_rd, y_proba_rd = evaluate_combo(
                X, labels, clf, combo_name, cv=cv_random, groups=None,
                cv_scheme="random",
            )
            results.append(result_rd)
            key_rd = f"{combo_name} | random"
            predictions_store["predictions"][key_rd] = y_proba_rd.tolist()
            predictions_store["thresholds"][key_rd]  = result_rd["optimal_threshold"]

    results_df = pd.DataFrame(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / base
    results_df.to_csv(f"{prefix}_classifier_comparison.csv", index=False)
    print(f"\n  ✅ Guardado: {prefix}_classifier_comparison.csv")

    with open(f"{prefix}_predictions.pkl", "wb") as f:
        pickle.dump(predictions_store, f)
    print(f"  ✅ Guardado: {prefix}_predictions.pkl  "
          f"(probabilidades out-of-fold de 9 combos × 2 esquemas CV, para Paso 7)")

    # ── Reporte: top 5 según el esquema PRIMARIO (espacial) ────────────────
    spatial_df = results_df[results_df["cv_scheme"] == "spatial"].sort_values(
        "f1_optimal", ascending=False
    )
    print(f"\n  📊 Top 5 combinaciones por F1 óptimo (CV espacial, métrica primaria):")
    print(spatial_df.head(5).to_string(index=False))

    best = spatial_df.iloc[0]
    best_random = results_df[
        (results_df["estrategia"] == best["estrategia"]) & (results_df["cv_scheme"] == "random")
    ].iloc[0]
    gap_auc = best_random["auc"] - best["auc"]

    print(f"\n  🏆 Mejor combinación (espacial): {best['estrategia']}")
    print(f"     AUC={best['auc']:.3f}   "
          f"F1(τ=0.5)={best['f1_default']:.3f}   "
          f"F1(τ_óptimo={best['optimal_threshold']:.3f})={best['f1_optimal']:.3f}")
    print(f"     Referencia random K-fold: AUC={best_random['auc']:.3f}  "
          f"(gap de leakage = {gap_auc:+.3f})")

    # ── Regla del proyecto: siempre usar el mejor FUSED, aunque otro gane global ──
    fused_df = spatial_df[spatial_df["estrategia"].str.startswith("fused + ")]
    best_fused = fused_df.iloc[0]
    if best_fused["estrategia"] != best["estrategia"]:
        delta_auc = best["auc"] - best_fused["auc"]
        delta_f1  = best["f1_optimal"] - best_fused["f1_optimal"]
        print(f"\n  🔗 Mejor FUSED (regla del proyecto, usar siempre este): {best_fused['estrategia']}")
        print(f"     AUC={best_fused['auc']:.3f}   F1_óptimo={best_fused['f1_optimal']:.3f}")
        print(f"     Costo de no usar el mejor global: ΔAUC=-{delta_auc:.3f}  ΔF1=-{delta_f1:.3f}")
    else:
        print(f"\n  🔗 El mejor FUSED ya coincide con el mejor global — sin trade-off.")

    print(f"\n  ✅ Listo — {distrito} {year_str}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Paso 6: clasificador final (umbral óptimo + no lineal + fusión)"
    )
    parser.add_argument("--distrito", default="all")
    parser.add_argument("--year", default="2016")
    parser.add_argument("--eigen-dir", default="eigen_features")
    parser.add_argument("--tensors-dir", default="neighbourhood_tensors")
    parser.add_argument("--graph-dir", default="graph_output")
    parser.add_argument("--output", default="final_classifier")
    parser.add_argument("--strat-tag", default="dualgmm")
    args = parser.parse_args()

    distritos = ["Barranco", "La Victoria"] if args.distrito == "all" else [args.distrito]
    years     = ([None] if args.year == "all" else [int(y) for y in args.year.split(",")])

    for distrito in distritos:
        for year in years:
            run_pipeline(
                distrito, year,
                Path(args.eigen_dir), Path(args.tensors_dir),
                Path(args.graph_dir), Path(args.output),
                strat_tag=args.strat_tag,
            )


if __name__ == "__main__":
    main()