"""
Paso 5: Features espectrales por vecindario (Laplaciano) — 2 estrategias
==========================================================================

Para cada vecindario tenemos una matriz X de forma (20, 768): 20 embeddings
de imágenes Street View (5 puntos de crimen × 4 imágenes c/u).

CORRECCIÓN IMPORTANTE sobre las dimensiones esperadas:
  El Laplaciano se construye sobre la matriz de SIMILITUD entre las 20
  imágenes (grafo de 20 nodos), no sobre el grafo de calles. Por lo tanto:
    - Sus eigenvectores son de dimensión 20 (uno por imagen), NO 512/768.
    - Top-K=5 eigenvectores → matriz (20, 5) → aplanada = 100 valores.
      (de aquí probablemente salía tu "100"). Esto captura solo la
      ESTRUCTURA de similitud entre imágenes, sin contenido visual.
    - Para capturar también el CONTENIDO visual, se proyectan los embeddings
      originales con los eigenvectores: U.T @ X → (5, 768) → aplanado=3840.
      Esta es la versión recomendada como feature principal (ver abajo).

ESTRATEGIA A — Laplaciano directo (768-dim):
  Distancias/similitud calculadas directamente sobre los embeddings crudos
  de 768 dimensiones. Riesgo: en alta dimensión las distancias coseno entre
  puntos tienden a concentrarse (curse of dimensionality), reduciendo el
  poder discriminativo del grafo de similitud.

ESTRATEGIA B — PCA global + Laplaciano:
  Se ajusta un PCA UNA SOLA VEZ sobre el pool completo de embeddings
  (las 27k imágenes, NO las 20 del vecindario — con solo 20 muestras no se
  puede estimar una covarianza de 768 dimensiones de forma estable). Luego
  se transforman las 20 embeddings del vecindario a la dimensión reducida
  (ej. 50) y se repite el mismo procedimiento de Laplaciano + eigenvectores.

COMPARACIÓN:
  Al final se entrena un clasificador rápido (regresión logística
  regularizada, cross-validada) sobre cada conjunto de features para dar
  una primera señal de cuál estrategia separa mejor peligroso/seguro.
  Esto es un DIAGNÓSTICO RÁPIDO, no el clasificador final del proyecto.

Uso:
    python Paso5_eigen_features.py \\
        --distrito Barranco --year 2016 \\
        --tensors-dir neighbourhood_tensors \\
        --embeddings-dir embeddings_export \\
        --graph-dir graph_output \\
        --strat-tag dualgmm \\
        --k 5 --pca-dim 50
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

K_EIGEN  = 5
PCA_DIM  = 50
SEED     = 42


# ─── Carga ─────────────────────────────────────────────────────────────────

def load_tensors(tensors_dir: Path, base: str):
    tensor = np.load(tensors_dir / f"{base}_neighbourhood_tensors.npy")
    with open(tensors_dir / f"{base}_node_ids_order.json") as f:
        node_ids = json.load(f)
    print(f"  Tensor cargado: shape {tensor.shape}  ({len(node_ids):,} vecindarios)")
    return tensor, node_ids


def load_labels(graph_dir: Path, base: str, strat_tag: str, node_ids: list) -> np.ndarray:
    """Carga el label primario (vecindario) desde nodes.csv, alineado al orden del tensor."""
    nodes_path = graph_dir / f"{base}_{strat_tag}_nodes.csv"
    nodes_df   = pd.read_csv(nodes_path).set_index("node_id")
    labels = []
    for n in node_ids:
        lbl = nodes_df.loc[n, "label"] if n in nodes_df.index else "seguro"
        labels.append(1 if lbl == "peligroso" else 0)
    labels = np.array(labels)
    print(f"  Labels cargados: {labels.sum():,} peligrosos / {len(labels):,} totales "
          f"({100*labels.mean():.1f}%)")
    return labels


def fit_global_pca(embeddings_dir: Path, pca_dim: int = PCA_DIM) -> PCA:
    """
    Ajusta PCA sobre el pool COMPLETO de embeddings (las 27k imágenes),
    no sobre las 20 del vecindario — crítico para estabilidad estadística.
    """
    all_embeddings = np.load(embeddings_dir / "embeddings.npy")
    print(f"  Ajustando PCA global: {all_embeddings.shape} → {pca_dim} dims…")
    pca = PCA(n_components=pca_dim, random_state=SEED)
    pca.fit(all_embeddings)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  Varianza explicada por {pca_dim} componentes: {100*var_explained:.1f}%")
    return pca


# ─── Laplaciano + eigenvectores ───────────────────────────────────────────────

def compute_laplacian_eigen_features(X: np.ndarray, k: int = K_EIGEN) -> dict:
    """
    Construye el grafo de similitud (coseno, no-negativo) entre las filas de X
    (20 imágenes), calcula el Laplaciano normalizado, y extrae:
      - eigvecs_flat     : top-k eigenvectores aplanados (k*20,) — estructura pura
      - eigen_projected  : U.T @ X aplanado (k*dim_X,) — estructura + contenido
                            (esta es la feature recomendada para el clasificador)

    NOTA: para el Laplaciano de grafos se usan las K eigenvalues MÁS PEQUEÑAS
    (excluyendo la trivial ~0), a diferencia de PCA donde se usan las más
    grandes. Es un punto de confusión común — aquí se hace explícito.
    """
    n = X.shape[0]

    # Similitud coseno (X puede o no estar normalizado L2 — se normaliza aquí
    # por seguridad, ya que tras un PCA los vectores pierden la norma unitaria)
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sim    = X_norm @ X_norm.T                      # (20,20), cos similarity
    W      = np.clip(sim, 0, None)                  # afinidad no-negativa
    np.fill_diagonal(W, 0)                          # sin auto-loops

    degree      = W.sum(axis=1)
    degree_safe = np.where(degree > 1e-9, degree, 1e-9)
    D_inv_sqrt  = np.diag(1.0 / np.sqrt(degree_safe))
    L_sym       = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt   # Laplaciano normalizado

    eigvals, eigvecs = np.linalg.eigh(L_sym)         # ascendente, simétrica real
    # Saltar el primer eigenvalue (trivial, ~0) y tomar los siguientes k
    start = 1 if n > k else 0
    idx   = np.arange(start, min(start + k, n))
    U     = eigvecs[:, idx]                          # (20, k)

    if U.shape[1] < k:   # padding si n es muy pequeño (no debería pasar, n=20 fijo)
        U = np.pad(U, ((0, 0), (0, k - U.shape[1])))

    eigvecs_flat    = U.flatten()                    # (k*20,) = 100 si k=5
    eigen_projected = (U.T @ X).flatten()             # (k*dim_X,)

    return {
        "eigvecs_flat":    eigvecs_flat,
        "eigen_projected": eigen_projected,
        "eigvals_used":    eigvals[idx],
    }


def compute_mean_pooling_features(tensor: np.ndarray, pca_model: PCA):
    """
    Baseline simple: promedio de las 20 embeddings por vecindario, sin
    Laplaciano. Sirve para verificar si la señal "cómo se ve este vecindario
    en promedio" se está perdiendo en el procedimiento espectral (al saltar
    el eigenvector trivial, que es ~proporcional al promedio ponderado).

      mean_raw : promedio de los 768-dim crudos      → (N, 768)
      mean_pca : promedio de los reducidos por PCA    → (N, pca_dim)
    """
    feats_mean_raw, feats_mean_pca = [], []
    for i in range(tensor.shape[0]):
        X     = tensor[i]                  # (20, 768)
        X_pca = pca_model.transform(X)     # (20, pca_dim)
        feats_mean_raw.append(X.mean(axis=0))
        feats_mean_pca.append(X_pca.mean(axis=0))
    return np.array(feats_mean_raw), np.array(feats_mean_pca)


def compute_mean_std_pooling_features(tensor: np.ndarray):
    """
    Mean + std concatenados (sin PCA, que ya demostró perder señal):
    captura no solo "cómo se ve en promedio" sino "cuánta diversidad visual
    hay" dentro del vecindario (heterogeneidad de fachadas, iluminación,
    densidad urbana, etc.) sin pagar el costo del Laplaciano completo.
    Shape: (N, 768*2) = (N, 1536)
    """
    feats = []
    for i in range(tensor.shape[0]):
        X = tensor[i]                       # (20, 768)
        feats.append(np.concatenate([X.mean(axis=0), X.std(axis=0)]))
    return np.array(feats)


def build_feature_matrices(tensor: np.ndarray, pca_model: PCA, k: int = K_EIGEN):
    """
    Para cada vecindario en el tensor, computa:
      - features_A: Laplaciano directo sobre los 768-dim crudos → eigen_projected (k*768,)
      - features_B: PCA global → Laplaciano sobre reducido → eigen_projected (k*pca_dim,)
    """
    feats_A, feats_B = [], []

    for i in range(tensor.shape[0]):
        X     = tensor[i]                    # (20, 768)
        X_pca = pca_model.transform(X)       # (20, pca_dim) — usa el PCA YA AJUSTADO

        res_A = compute_laplacian_eigen_features(X, k=k)
        res_B = compute_laplacian_eigen_features(X_pca, k=k)

        feats_A.append(res_A["eigen_projected"])
        feats_B.append(res_B["eigen_projected"])

    return np.array(feats_A), np.array(feats_B)


# ─── Comparación rápida (diagnóstico, no clasificador final) ────────────────

def quick_classifier_comparison(
    features_dict: dict,
    labels: np.ndarray,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """
    Entrena una regresión logística regularizada con CV REPETIDA (n_repeats×5
    folds) sobre cada conjunto de features y compara F1/AUC con su desviación
    estándar entre repeticiones. Con N~400, un solo 5-fold tiene varianza
    muestral notable; repetir con distintos splits da una estimación más
    confiable de si la diferencia entre estrategias es real o ruido.

    Esto sigue siendo un DIAGNÓSTICO RÁPIDO, no el clasificador final.
    """
    n_minority = min(labels.sum(), len(labels) - labels.sum())
    n_splits   = min(5, max(2, int(n_minority)))

    results = []
    for name, X in features_dict.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=2000,
                                       class_weight="balanced", random_state=SEED)),
        ])
        try:
            cv = RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=SEED
            )
            scores = cross_validate(
                pipe, X, labels, cv=cv,
                scoring=["accuracy", "f1", "roc_auc"],
            )
            results.append({
                "estrategia":    name,
                "n_features":    X.shape[1],
                "accuracy_mean": scores["test_accuracy"].mean(),
                "f1_mean":       scores["test_f1"].mean(),
                "auc_mean":      scores["test_roc_auc"].mean(),
                "auc_std":       scores["test_roc_auc"].std(),
                "n_splits":      n_splits,
                "n_repeats":     n_repeats,
            })
        except Exception as e:
            print(f"  ⚠ No se pudo evaluar '{name}': {e}")

    df = pd.DataFrame(results)
    print(f"\n  📊 Comparación rápida ({n_splits}-fold × {n_repeats} repeticiones, "
          f"diagnóstico no definitivo):")
    print(df.to_string(index=False))
    return df


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_pipeline(
    distrito: str,
    year,
    tensors_dir: Path,
    embeddings_dir: Path,
    graph_dir: Path,
    output_dir: Path,
    strat_tag: str = "dualgmm",
    k: int = K_EIGEN,
    pca_dim: int = PCA_DIM,
):
    year_str = str(year) if year else "all"
    base     = f"{distrito.replace(' ', '_')}_{year_str}"
    print(f"\n{'='*60}")
    print(f"  Distrito: {distrito} | Año: {year_str}")
    print(f"{'='*60}")

    tensor, node_ids = load_tensors(tensors_dir, base)
    labels            = load_labels(graph_dir, base, strat_tag, node_ids)
    pca_model         = fit_global_pca(embeddings_dir, pca_dim)

    print(f"\n  Calculando eigen-features para {tensor.shape[0]:,} vecindarios…")
    features_A, features_B = build_feature_matrices(tensor, pca_model, k=k)

    print(f"  Calculando baseline de mean-pooling (sin Laplaciano)…")
    features_mean_raw, features_mean_pca = compute_mean_pooling_features(tensor, pca_model)
    features_mean_std = compute_mean_std_pooling_features(tensor)

    print(f"  Laplaciano directo (768-dim) : {features_A.shape}")
    print(f"  PCA + Laplaciano             : {features_B.shape}")
    print(f"  Mean-pooling directo (768)   : {features_mean_raw.shape}")
    print(f"  Mean-pooling + PCA           : {features_mean_pca.shape}")
    print(f"  Mean+Std pooling (1536)      : {features_mean_std.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / base
    np.save(f"{prefix}_features_laplacian_raw.npy", features_A)
    np.save(f"{prefix}_features_pca_laplacian.npy", features_B)
    np.save(f"{prefix}_features_mean_raw.npy", features_mean_raw)
    np.save(f"{prefix}_features_mean_pca.npy", features_mean_pca)
    np.save(f"{prefix}_features_mean_std.npy", features_mean_std)
    np.save(f"{prefix}_labels.npy", labels)
    print(f"\n  ✅ Guardado: {prefix}_features_*.npy  (5 estrategias) + labels.npy")

    features_dict = {
        "Laplaciano directo (768-dim)": features_A,
        "PCA + Laplaciano":             features_B,
        "Mean-pooling directo (768)":   features_mean_raw,
        "Mean-pooling + PCA":           features_mean_pca,
        "Mean+Std pooling (1536)":      features_mean_std,
    }
    comparison_df = quick_classifier_comparison(features_dict, labels)
    comparison_df.to_csv(f"{prefix}_strategy_comparison.csv", index=False)
    print(f"\n  ✅ Guardado: {prefix}_strategy_comparison.csv")

    print(f"\n  ✅ Listo — {distrito} {year_str}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paso 5: Eigen-features (Laplaciano) por vecindario")
    parser.add_argument("--distrito", default="all")
    parser.add_argument("--year", default="2016")
    parser.add_argument("--tensors-dir", default="neighbourhood_tensors")
    parser.add_argument("--embeddings-dir", default="embeddings_export")
    parser.add_argument("--graph-dir", default="graph_output")
    parser.add_argument("--output", default="eigen_features")
    parser.add_argument("--strat-tag", default="dualgmm")
    parser.add_argument("--k", type=int, default=K_EIGEN, help="Número de eigenvectores (top-K)")
    parser.add_argument("--pca-dim", type=int, default=PCA_DIM, help="Dimensión del PCA global")

    args = parser.parse_args()

    distritos = ["Barranco", "La Victoria"] if args.distrito == "all" else [args.distrito]
    years     = ([None] if args.year == "all" else [int(y) for y in args.year.split(",")])

    for distrito in distritos:
        for year in years:
            run_pipeline(
                distrito, year,
                Path(args.tensors_dir), Path(args.embeddings_dir),
                Path(args.graph_dir), Path(args.output),
                strat_tag=args.strat_tag, k=args.k, pca_dim=args.pca_dim,
            )


if __name__ == "__main__":
    main()