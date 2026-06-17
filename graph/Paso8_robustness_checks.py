"""
Paso 8: Validaciones de robustez del clasificador final
===========================================================

Tres chequeos, en un solo script porque comparten la misma data de entrada:

  ① LEAKAGE ESPACIAL
     Los vecindarios son ego-graphs que se solapan: dos nodos adyacentes
     comparten buena parte de los mismos puntos de crimen (y por tanto las
     mismas imágenes) en sus tensores de 20 embeddings. Con K-fold aleatorio,
     es muy probable que un nodo y su vecino casi-idéntico terminen en folds
     distintos — el modelo "ve" en entrenamiento información casi igual a la
     de test, inflando el AUC reportado.

     Fix: GroupKFold con bloques GEOGRÁFICOS (KMeans sobre lat/lon, 5 bloques).
     Todos los nodos de un mismo bloque caen siempre en el mismo fold, así
     nunca se entrena con el vecino casi-idéntico del nodo de test. Si el AUC
     bajo este esquema es similar al de K-fold aleatorio, el resultado es
     robusto. Si cae mucho, había leakage.

  ② ESTABILIDAD DEL UMBRAL ÓPTIMO
     El umbral óptimo de F1 (0.36 en Barranco, 0.33 en La Victoria) se
     calculó con UN solo split de 5-fold. Se repite con 10 seeds distintos
     de partición para ver si el umbral es estable o se mueve mucho.

  ③ STACKING (ensamble de 2 niveles)
     En vez de concatenar features visual+estructural (fusión simple), se
     entrena un modelo separado por modalidad y un meta-clasificador que
     combina sus probabilidades de salida. sklearn's StackingClassifier ya
     maneja la validación cruzada anidada internamente (evita leakage del
     propio stacking).

REQUISITO PREVIO: Paso 4, 5 y 6 ya corridos (reusa sus outputs + las
funciones de Paso6_final_classifier.py, debe estar en la misma carpeta).

Uso:
    python Paso8_robustness_checks.py --distrito all --year 2016
"""

import argparse
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Paso6_final_classifier import (compute_structural_features, rebuild_graph,
                                    find_optimal_threshold_f1)

warnings.filterwarnings("ignore")

SEED              = 42
N_SPLITS          = 5
N_SPATIAL_BLOCKS  = 5
N_THRESHOLD_SEEDS = 10


def _logistic():
    return LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=SEED)


# ─── ① Leakage espacial ────────────────────────────────────────────────────

def assign_spatial_blocks(nodes_csv: Path, node_ids: list, n_blocks: int = N_SPATIAL_BLOCKS):
    nodes_df = pd.read_csv(nodes_csv).set_index("node_id")
    coords = np.array([[nodes_df.loc[n, "lon"], nodes_df.loc[n, "lat"]] for n in node_ids])
    km = KMeans(n_clusters=n_blocks, random_state=SEED, n_init=10).fit(coords)
    return km.labels_


def evaluate_random_cv(X, y, clf, name):
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    cv   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    y_proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, y_proba)
    opt_thr, f1_opt = find_optimal_threshold_f1(y, y_proba)
    return {"estrategia": name, "cv_scheme": "random_kfold", "auc": auc,
            "f1_default": f1_score(y, (y_proba >= 0.5).astype(int)),
            "optimal_threshold": opt_thr, "f1_optimal": f1_opt}


def evaluate_spatial_cv(X, y, groups, clf, name):
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    cv   = GroupKFold(n_splits=N_SPATIAL_BLOCKS)
    y_proba = cross_val_predict(pipe, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, y_proba)
    opt_thr, f1_opt = find_optimal_threshold_f1(y, y_proba)
    return {"estrategia": name, "cv_scheme": "spatial_blocks", "auc": auc,
            "f1_default": f1_score(y, (y_proba >= 0.5).astype(int)),
            "optimal_threshold": opt_thr, "f1_optimal": f1_opt}


# ─── ② Estabilidad del umbral ──────────────────────────────────────────────

def evaluate_threshold_stability(X, y, clf, name, n_seeds: int = N_THRESHOLD_SEEDS):
    thresholds, f1s, aucs = [], [], []
    for seed in range(n_seeds):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        cv   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        y_proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        opt_thr, f1_opt = find_optimal_threshold_f1(y, y_proba)
        thresholds.append(opt_thr)
        f1s.append(f1_opt)
        aucs.append(roc_auc_score(y, y_proba))
    return {
        "estrategia": name, "n_seeds": n_seeds,
        "threshold_mean": float(np.mean(thresholds)), "threshold_std": float(np.std(thresholds)),
        "f1_optimal_mean": float(np.mean(f1s)), "f1_optimal_std": float(np.std(f1s)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
    }


# ─── ③ Stacking ────────────────────────────────────────────────────────────

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selecciona un rango de columnas — usado para que cada base estimator
    del stacking solo vea su propia modalidad (visual o estructural)."""
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.start:self.end]


def build_stacking_classifier(n_visual: int, n_structural: int, base_kind: str = "logistic"):
    if base_kind == "logistic":
        base_visual     = _logistic()
        base_structural = _logistic()
    else:
        base_visual = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=SEED, n_jobs=-1,
        )
        base_structural = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=SEED, n_jobs=-1,
        )

    visual_pipe = Pipeline([
        ("select", ColumnSelector(0, n_visual)),
        ("scaler", StandardScaler()),
        ("clf", base_visual),
    ])
    structural_pipe = Pipeline([
        ("select", ColumnSelector(n_visual, n_visual + n_structural)),
        ("scaler", StandardScaler()),
        ("clf", base_structural),
    ])

    return StackingClassifier(
        estimators=[("visual", visual_pipe), ("structural", structural_pipe)],
        final_estimator=LogisticRegression(class_weight="balanced", random_state=SEED),
        cv=5, stack_method="predict_proba", n_jobs=-1,
    )


def evaluate_stacking(fused, y, n_visual, n_structural, base_kind, name):
    stacking = build_stacking_classifier(n_visual, n_structural, base_kind)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    y_proba = cross_val_predict(stacking, fused, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, y_proba)
    opt_thr, f1_opt = find_optimal_threshold_f1(y, y_proba)
    return {"estrategia": name, "auc": auc,
            "f1_default": f1_score(y, (y_proba >= 0.5).astype(int)),
            "optimal_threshold": opt_thr, "f1_optimal": f1_opt}


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_pipeline(
    distrito: str, year, eigen_dir: Path, tensors_dir: Path, graph_dir: Path,
    output_dir: Path, strat_tag: str = "dualgmm", stacking_base: str = "logistic",
):
    year_str = str(year) if year else "all"
    base     = f"{distrito.replace(' ', '_')}_{year_str}"
    print(f"\n{'='*60}\n  Distrito: {distrito} | Año: {year_str}\n{'='*60}")

    visual = np.load(eigen_dir / f"{base}_features_mean_std.npy")
    labels = np.load(eigen_dir / f"{base}_labels.npy")
    with open(tensors_dir / f"{base}_node_ids_order.json") as f:
        node_ids = json.load(f)

    nodes_csv = graph_dir / f"{base}_{strat_tag}_nodes.csv"
    G, _ = rebuild_graph(nodes_csv, graph_dir / f"{base}_edges.csv")
    structural, structural_names = compute_structural_features(G, node_ids)
    fused = np.hstack([visual, structural])
    n_visual, n_structural = visual.shape[1], structural.shape[1]

    results = []

    # ── ① Leakage espacial ──────────────────────────────────────────────────
    print("\n  ① Leakage espacial: random K-fold vs bloques geográficos")
    random_result = evaluate_random_cv(fused, labels, _logistic(), "fused + logistic")
    results.append(random_result)

    groups = assign_spatial_blocks(nodes_csv, node_ids)
    block_sizes = pd.Series(groups).value_counts().sort_index().tolist()
    print(f"     Tamaño de los 5 bloques geográficos: {block_sizes}")

    spatial_result = evaluate_spatial_cv(fused, labels, groups, _logistic(), "fused + logistic")
    results.append(spatial_result)

    gap_auc = random_result["auc"] - spatial_result["auc"]
    gap_f1  = random_result["f1_optimal"] - spatial_result["f1_optimal"]
    print(f"     AUC random K-fold   = {random_result['auc']:.3f}   "
          f"F1_óptimo = {random_result['f1_optimal']:.3f}")
    print(f"     AUC bloques espaciales = {spatial_result['auc']:.3f}   "
          f"F1_óptimo = {spatial_result['f1_optimal']:.3f}")
    print(f"     Gap (random − espacial): ΔAUC={gap_auc:+.3f}  ΔF1={gap_f1:+.3f}")
    if gap_auc > 0.05:
        print(f"     ⚠️  Gap de AUC >0.05 — hay evidencia de leakage espacial significativo.")
    else:
        print(f"     ✅ Gap pequeño — el resultado es razonablemente robusto a la correlación espacial.")

    # ── ② Estabilidad del umbral ────────────────────────────────────────────
    print(f"\n  ② Estabilidad del umbral óptimo ({N_THRESHOLD_SEEDS} seeds de partición)")
    stability = evaluate_threshold_stability(fused, labels, _logistic(), "fused + logistic")
    print(f"     τ_óptimo = {stability['threshold_mean']:.3f} ± {stability['threshold_std']:.3f}")
    print(f"     F1_óptimo = {stability['f1_optimal_mean']:.3f} ± {stability['f1_optimal_std']:.3f}")
    print(f"     AUC = {stability['auc_mean']:.3f} ± {stability['auc_std']:.3f}")
    if stability["threshold_std"] > 0.05:
        print(f"     ⚠️  El umbral varía bastante entre particiones (std>0.05) — "
              f"repórtalo como rango, no como valor puntual.")
    else:
        print(f"     ✅ Umbral estable entre particiones.")

    # ── ③ Stacking ───────────────────────────────────────────────────────────
    print(f"\n  ③ Stacking ({stacking_base}) vs fusión simple")
    stacking_result = evaluate_stacking(fused, labels, n_visual, n_structural,
                                        stacking_base, f"stacking ({stacking_base})")
    results.append(stacking_result)
    delta_auc_stack = stacking_result["auc"] - random_result["auc"]
    delta_f1_stack  = stacking_result["f1_optimal"] - random_result["f1_optimal"]
    print(f"     AUC stacking = {stacking_result['auc']:.3f}  "
          f"(vs fusión simple {random_result['auc']:.3f}, Δ={delta_auc_stack:+.3f})")
    print(f"     F1_óptimo stacking = {stacking_result['f1_optimal']:.3f}  "
          f"(vs fusión simple {random_result['f1_optimal']:.3f}, Δ={delta_f1_stack:+.3f})")

    # ── Guardar ──────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / base

    pd.DataFrame(results).to_csv(f"{prefix}_robustness_comparison.csv", index=False)
    with open(f"{prefix}_threshold_stability.json", "w") as f:
        json.dump(stability, f, indent=2)
    print(f"\n  ✅ Guardado: {prefix}_robustness_comparison.csv")
    print(f"  ✅ Guardado: {prefix}_threshold_stability.json")
    print(f"\n  ✅ Listo — {distrito} {year_str}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paso 8: validaciones de robustez")
    parser.add_argument("--distrito", default="all")
    parser.add_argument("--year", default="2016")
    parser.add_argument("--eigen-dir", default="eigen_features")
    parser.add_argument("--tensors-dir", default="neighbourhood_tensors")
    parser.add_argument("--graph-dir", default="graph_output")
    parser.add_argument("--output", default="robustness_checks")
    parser.add_argument("--strat-tag", default="dualgmm")
    parser.add_argument("--stacking-base", default="logistic", choices=["logistic", "random_forest"])
    args = parser.parse_args()

    distritos = ["Barranco", "La Victoria"] if args.distrito == "all" else [args.distrito]
    years     = ([None] if args.year == "all" else [int(y) for y in args.year.split(",")])

    for distrito in distritos:
        for year in years:
            run_pipeline(
                distrito, year,
                Path(args.eigen_dir), Path(args.tensors_dir), Path(args.graph_dir),
                Path(args.output), strat_tag=args.strat_tag, stacking_base=args.stacking_base,
            )


if __name__ == "__main__":
    main()