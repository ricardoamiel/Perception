"""
Paso 7: Plots diagnósticos para entender los resultados
==========================================================

Genera 6 visualizaciones a partir de los outputs de Paso 6:

  1. metric_vs_threshold     — precision/recall/F1 vs umbral, con el óptimo marcado
  2. roc_pr_comparison       — ROC y PR overlay: visual vs estructural vs fusionado
  3. confusion_matrix        — matriz de confusión al umbral óptimo
  4. graph_real_vs_predicted — grafo de calles, 2 paneles: label real vs predicho
  5. graph_error_map         — grafo de calles, 1 panel: TP/TN/FP/FN coloreados
  6. feature_group_importance— Random Forest sobre fusionado: cuánto aporta
                                visual vs estructural (importancias agrupadas)

Todos en fondo blanco/vanilla, consistente con el resto del pipeline.

REQUISITO PREVIO: Paso 6 ya corrido (usa su predictions.pkl directamente).

Uso:
    python Paso7_diagnostic_plots.py --distrito Barranco --year 2016
    python Paso7_diagnostic_plots.py --distrito "La Victoria" --year 2016 --combo "fused + random_forest"
"""

import argparse
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                              roc_curve, auc as sk_auc)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Paleta vanilla (consistente con graph_pipeline.py) ───────────────────────

BG          = "#fafaf7"
EDGE_COLOR  = "#b0b8c8"
TEXT_COLOR  = "#475569"
GRAY        = "#9ca3af"
RED         = "#dc2626"     # peligroso / falso negativo (el error más grave)
GREEN       = "#16a34a"     # seguro / verdadero positivo
ORANGE      = "#f59e0b"     # falso positivo (falsa alarma)
BLUE        = "#1d4ed8"


# ─── Carga ─────────────────────────────────────────────────────────────────

def load_predictions(final_dir: Path, base: str) -> dict:
    with open(final_dir / f"{base}_predictions.pkl", "rb") as f:
        return pickle.load(f)


def load_comparison_table(final_dir: Path, base: str) -> pd.DataFrame:
    return pd.read_csv(final_dir / f"{base}_classifier_comparison.csv")


def rebuild_graph(nodes_csv: Path, edges_csv: Path):
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        G.add_node(int(row["node_id"]), x=row["lon"], y=row["lat"])
    for _, row in edges_df.iterrows():
        G.add_edge(int(row["u"]), int(row["v"]))
    return G


def _draw_edges(ax, G):
    segments = [
        ((G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"]))
        for u, v in G.edges()
    ]
    lc = LineCollection(segments, colors=EDGE_COLOR, linewidths=0.5,
                         alpha=0.8, zorder=1)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")


# ─── Plot 1: métrica vs umbral ────────────────────────────────────────────────

def plot_metric_vs_threshold(y_true, y_proba, combo_name, optimal_threshold,
                              output_path: Path):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    ax.set_facecolor(BG)
    ax.plot(thresholds, precisions[:-1], color=BLUE, linewidth=1.8, label="Precision")
    ax.plot(thresholds, recalls[:-1], color="#d97706", linewidth=1.8, label="Recall")
    ax.plot(thresholds, f1s[:-1], color=RED, linewidth=2.2, label="F1")
    ax.axvline(optimal_threshold, color=TEXT_COLOR, linestyle="--", linewidth=1.2,
               label=f"Umbral óptimo = {optimal_threshold:.2f}")
    ax.axvline(0.5, color=GRAY, linestyle=":", linewidth=1.0, label="Umbral default = 0.50")

    ax.set_xlabel("Umbral de decisión")
    ax.set_ylabel("Métrica")
    ax.set_title(f"Precision / Recall / F1 vs umbral\n{combo_name}",
                color=TEXT_COLOR, fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, color="#e2e8f0", linewidth=0.6)
    ax.legend(fontsize=9, facecolor="white", edgecolor="#cbd5e1", labelcolor=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Guardado: {output_path}")


# ─── Plot 2: ROC + PR comparando feature sets ────────────────────────────────

def plot_roc_pr_comparison(predictions: dict, y_true, classifier_suffix, cv_scheme,
                            output_path: Path):
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
    colors = {"visual_only": "#d85a30", "structural_only": "#888780",
              "fused": "#1d9e75"}
    labels_es = {"visual_only": "Solo visual", "structural_only": "Solo estructural",
                 "fused": "Fusionado"}

    for ax in (ax_roc, ax_pr):
        ax.set_facecolor(BG)
        ax.grid(True, color="#e2e8f0", linewidth=0.6)
        ax.tick_params(colors=TEXT_COLOR)

    for fs_name in ["visual_only", "structural_only", "fused"]:
        combo = f"{fs_name} + {classifier_suffix} | {cv_scheme}"
        if combo not in predictions["predictions"]:
            continue
        y_proba = np.array(predictions["predictions"][combo])

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc_val = sk_auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=colors[fs_name], linewidth=2,
                    label=f"{labels_es[fs_name]} (AUC={roc_auc_val:.2f})")

        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        ax_pr.plot(rec, prec, color=colors[fs_name], linewidth=2,
                   label=labels_es[fs_name])

    cv_label = "CV espacial — sin leakage" if cv_scheme == "spatial" else "CV aleatorio — referencia"
    ax_roc.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1, label="Azar")
    ax_roc.set_xlabel("Tasa de falsos positivos")
    ax_roc.set_ylabel("Tasa de verdaderos positivos")
    ax_roc.set_title(f"Curva ROC ({classifier_suffix})\n{cv_label}", color=TEXT_COLOR, fontsize=11)
    ax_roc.legend(fontsize=8.5, facecolor="white", edgecolor="#cbd5e1", labelcolor=TEXT_COLOR)

    base_rate = y_true.mean()
    ax_pr.axhline(base_rate, color=GRAY, linestyle="--", linewidth=1,
                  label=f"Azar (tasa base={base_rate:.2f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Curva Precision-Recall ({classifier_suffix})\n{cv_label}", color=TEXT_COLOR, fontsize=11)
    ax_pr.legend(fontsize=8.5, facecolor="white", edgecolor="#cbd5e1", labelcolor=TEXT_COLOR)

    for ax in (ax_roc, ax_pr):
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Guardado: {output_path}")


# ─── Plot 3: matriz de confusión ──────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_proba, threshold, combo_name, output_path: Path):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5.5, 5), facecolor="white")
    im = ax.imshow(cm, cmap="Reds", alpha=0.85)

    labels = ["seguro", "peligroso"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicho", color=TEXT_COLOR)
    ax.set_ylabel("Real", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.set_title(f"Matriz de confusión (τ={threshold:.2f})\n{combo_name}",
                color=TEXT_COLOR, fontsize=11)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else TEXT_COLOR
            ax.text(j, i, str(val), ha="center", va="center",
                   color=color, fontsize=16, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Guardado: {output_path}")


# ─── Plot 4: grafo real vs predicho ───────────────────────────────────────────

def plot_graph_real_vs_predicted(G, node_ids, y_true, y_proba, threshold,
                                  distrito, year_str, cv_scheme, output_path: Path):
    y_pred = (y_proba >= threshold).astype(int)

    fig, (ax_real, ax_pred) = plt.subplots(1, 2, figsize=(14, 7), facecolor="white")
    for ax, values, title in [
        (ax_real, y_true, "Label real (GMM)"),
        (ax_pred, y_pred, "Label predicho (modelo)"),
    ]:
        ax.set_facecolor(BG)
        _draw_edges(ax, G)
        colors = [RED if v == 1 else GREEN for v in values]
        xs = [G.nodes[n]["x"] for n in node_ids]
        ys = [G.nodes[n]["y"] for n in node_ids]
        ax.scatter(xs, ys, c=colors, s=14, zorder=2, linewidths=0)
        ax.set_title(title, color=TEXT_COLOR, fontsize=12)

    patches = [
        mpatches.Patch(color=RED, label="Peligroso"),
        mpatches.Patch(color=GREEN, label="Seguro"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=10,
              facecolor="white", edgecolor="#cbd5e1", labelcolor=TEXT_COLOR,
              bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Real vs predicho — {distrito} {year_str}  (CV {cv_scheme})",
                color=TEXT_COLOR, fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Guardado: {output_path}")


# ─── Plot 5: mapa de errores (TP/TN/FP/FN) ───────────────────────────────────

def plot_graph_error_map(G, node_ids, y_true, y_proba, threshold,
                          distrito, year_str, cv_scheme, output_path: Path):
    y_pred = (y_proba >= threshold).astype(int)

    categories = []
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            categories.append("TP")
        elif yt == 0 and yp == 0:
            categories.append("TN")
        elif yt == 0 and yp == 1:
            categories.append("FP")
        else:
            categories.append("FN")

    color_map = {"TP": "#991b1b", "TN": GRAY, "FP": ORANGE, "FN": RED}
    size_map  = {"TP": 16, "TN": 8, "FP": 16, "FN": 22}   # FN más grande: el error más grave

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    ax.set_facecolor(BG)
    _draw_edges(ax, G)

    xs = [G.nodes[n]["x"] for n in node_ids]
    ys = [G.nodes[n]["y"] for n in node_ids]
    colors = [color_map[c] for c in categories]
    sizes  = [size_map[c] for c in categories]
    ax.scatter(xs, ys, c=colors, s=sizes, zorder=2, linewidths=0)

    patches = [
        mpatches.Patch(color="#991b1b", label="Verdadero peligroso (TP)"),
        mpatches.Patch(color=GRAY, label="Verdadero seguro (TN)"),
        mpatches.Patch(color=ORANGE, label="Falsa alarma (FP)"),
        mpatches.Patch(color=RED, label="Peligro no detectado (FN)"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9,
             facecolor="white", edgecolor="#cbd5e1", labelcolor=TEXT_COLOR)

    n_fn = categories.count("FN")
    ax.set_title(
        f"Mapa de errores del clasificador — {distrito} {year_str}  (CV {cv_scheme})\n"
        f"{n_fn} vecindarios peligrosos NO detectados (FN, en rojo intenso)",
        color=TEXT_COLOR, fontsize=12,
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Guardado: {output_path}")


# ─── Plot 6: importancia de features agrupada (visual vs estructural) ───────

def plot_feature_group_importance(visual, structural, structural_names,
                                  y_true, distrito, year_str, output_path: Path):
    fused = np.hstack([visual, structural])
    scaler = StandardScaler()
    fused_scaled = scaler.fit_transform(fused)

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(fused_scaled, y_true)
    importances = rf.feature_importances_

    n_visual = visual.shape[1]
    visual_importance_total     = importances[:n_visual].sum()
    structural_importance_total = importances[n_visual:].sum()

    fig, (ax_group, ax_struct) = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    for ax in (ax_group, ax_struct):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT_COLOR)

    # Panel izquierdo: importancia total visual vs estructural
    ax_group.bar(["Visual\n(1536 dims)", "Estructural\n(5 dims)"],
                [visual_importance_total, structural_importance_total],
                color=["#d85a30", "#1d9e75"])
    ax_group.set_ylabel("Importancia total (suma)", color=TEXT_COLOR)
    ax_group.set_title("Aporte agregado: visual vs estructural",
                       color=TEXT_COLOR, fontsize=11)
    for i, v in enumerate([visual_importance_total, structural_importance_total]):
        ax_group.text(i, v + 0.01, f"{v:.2f}", ha="center", color=TEXT_COLOR, fontsize=10)

    # Panel derecho: detalle de cada feature estructural individual
    struct_importances = importances[n_visual:]
    order = np.argsort(struct_importances)[::-1]
    ax_struct.barh(
        [structural_names[i] for i in order][::-1],
        [struct_importances[i] for i in order][::-1],
        color="#1d9e75",
    )
    ax_struct.set_xlabel("Importancia individual", color=TEXT_COLOR)
    ax_struct.set_title("Detalle: features estructurales", color=TEXT_COLOR, fontsize=11)

    for ax in (ax_group, ax_struct):
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)

    fig.suptitle(f"Importancia de features (Random Forest) — {distrito} {year_str}",
                color=TEXT_COLOR, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Guardado: {output_path}")


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_pipeline(
    distrito: str,
    year,
    final_dir: Path,
    eigen_dir: Path,
    graph_dir: Path,
    output_dir: Path,
    combo: str = None,
    cv_scheme: str = "spatial",
    feature_set: str = "fused",
    strat_tag: str = "dualgmm",
):
    year_str = str(year) if year else "all"
    base     = f"{distrito.replace(' ', '_')}_{year_str}"
    print(f"\n{'='*60}")
    print(f"  Distrito: {distrito} | Año: {year_str}")
    print(f"  Esquema CV: {cv_scheme}  "
          f"({'sin leakage, métrica primaria' if cv_scheme == 'spatial' else 'referencia, infla resultados'})")
    print(f"{'='*60}")

    predictions = load_predictions(final_dir, base)
    comparison  = load_comparison_table(final_dir, base)

    scheme_df = comparison[comparison["cv_scheme"] == cv_scheme]

    if combo is None:
        if feature_set != "any":
            # Regla del proyecto: siempre usar el mejor combo cuyo feature_set
            # sea `feature_set` (default "fused"), aunque otro feature_set
            # tenga mejor F1 en abstracto. Se imprime también el mejor global
            # para que el costo de esta decisión quede explícito en consola.
            filtered_df = scheme_df[scheme_df["estrategia"].str.startswith(f"{feature_set} + ")]
            if filtered_df.empty:
                raise ValueError(f"No hay combinaciones con feature_set='{feature_set}' "
                                 f"en {base} bajo cv_scheme='{cv_scheme}'.")
            combo = filtered_df.sort_values("f1_optimal", ascending=False).iloc[0]["estrategia"]

            global_best = scheme_df.sort_values("f1_optimal", ascending=False).iloc[0]
            if global_best["estrategia"] != combo:
                chosen_row = filtered_df[filtered_df["estrategia"] == combo].iloc[0]
                print(f"  ℹ Mejor global sería '{global_best['estrategia']}' "
                      f"(AUC={global_best['auc']:.3f}, F1={global_best['f1_optimal']:.3f}),")
                print(f"    pero la regla del proyecto exige feature_set='{feature_set}' → "
                      f"se usa '{combo}' (AUC={chosen_row['auc']:.3f}, F1={chosen_row['f1_optimal']:.3f})")
        else:
            combo = scheme_df.sort_values("f1_optimal", ascending=False).iloc[0]["estrategia"]

    print(f"  Combinación elegida: {combo}  | {cv_scheme}")

    pred_key  = f"{combo} | {cv_scheme}"
    y_true    = np.array(predictions["y_true"])
    y_proba   = np.array(predictions["predictions"][pred_key])
    threshold = predictions["thresholds"][pred_key]
    classifier_suffix = combo.split(" + ")[1]
    combo_label = f"{combo}\n(CV {cv_scheme})"

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / base

    plot_metric_vs_threshold(y_true, y_proba, combo_label, threshold,
                             f"{prefix}_metric_vs_threshold.png")

    plot_roc_pr_comparison(predictions, y_true, classifier_suffix, cv_scheme,
                           f"{prefix}_roc_pr_comparison.png")

    plot_confusion_matrix(y_true, y_proba, threshold, combo_label,
                          f"{prefix}_confusion_matrix.png")

    G = rebuild_graph(
        graph_dir / f"{base}_{strat_tag}_nodes.csv",
        graph_dir / f"{base}_edges.csv",
    )
    node_ids = predictions["node_ids"]

    plot_graph_real_vs_predicted(G, node_ids, y_true, y_proba, threshold,
                                 distrito, year_str, cv_scheme,
                                 f"{prefix}_graph_real_vs_predicted.png")

    plot_graph_error_map(G, node_ids, y_true, y_proba, threshold,
                         distrito, year_str, cv_scheme,
                         f"{prefix}_graph_error_map.png")

    visual = np.load(eigen_dir / f"{base}_features_mean_std.npy")
    structural_names = predictions["structural_feature_names"]
    from Paso6_final_classifier import compute_structural_features
    structural, _ = compute_structural_features(G, node_ids)

    plot_feature_group_importance(visual, structural, structural_names, y_true,
                                  distrito, year_str,
                                  f"{prefix}_feature_group_importance.png")

    print(f"\n  ✅ Listo — {distrito} {year_str}  (6 plots guardados en {output_dir})")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paso 7: plots diagnósticos")
    parser.add_argument("--distrito", default="all")
    parser.add_argument("--year", default="2016")
    parser.add_argument("--final-dir", default="final_classifier")
    parser.add_argument("--eigen-dir", default="eigen_features")
    parser.add_argument("--graph-dir", default="graph_output")
    parser.add_argument("--output", default="diagnostic_plots")
    parser.add_argument("--strat-tag", default="dualgmm")
    parser.add_argument("--cv-scheme", default="spatial", choices=["spatial", "random"],
                        help="Qué esquema de CV usar para elegir/graficar la combinación "
                             "(default 'spatial', el que evita leakage)")
    parser.add_argument("--feature-set", default="fused",
                        choices=["fused", "visual_only", "structural_only", "any"],
                        help="Regla de selección: 'fused' (default, regla del proyecto) "
                             "fuerza usar el mejor combo fusionado aunque otro feature_set "
                             "tenga mejor F1; 'any' usa el mejor global sin restricción.")
    parser.add_argument("--combo", default=None,
                        help="Combinación exacta a graficar (ej. 'fused + random_forest'). "
                             "Si se especifica, ignora --feature-set.")
    args = parser.parse_args()

    distritos = ["Barranco", "La Victoria"] if args.distrito == "all" else [args.distrito]
    years     = ([None] if args.year == "all" else [int(y) for y in args.year.split(",")])

    for distrito in distritos:
        for year in years:
            run_pipeline(
                distrito, year,
                Path(args.final_dir), Path(args.eigen_dir),
                Path(args.graph_dir), Path(args.output),
                combo=args.combo, cv_scheme=args.cv_scheme,
                feature_set=args.feature_set, strat_tag=args.strat_tag,
            )


if __name__ == "__main__":
    main()