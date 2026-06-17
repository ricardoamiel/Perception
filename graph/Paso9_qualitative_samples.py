"""
Paso 9: Muestras cualitativas — imágenes reales de vecindarios aleatorios
=============================================================================

Genera 3 grids de imágenes Street View REALES (no embeddings, las fotos en sí)
para inspección cualitativa:

  1. Barranco solamente   — N peligrosos + N seguros, elegidos al azar
  2. La Victoria solamente — N peligrosos + N seguros, elegidos al azar
  3. Combinado             — 4 filas: Barranco-peligroso, Barranco-seguro,
                              La_Victoria-peligroso, La_Victoria-seguro,
                              para comparar visualmente si el "look" del
                              riesgo se parece entre distritos o no

Cada imagen se bordea en rojo (peligroso) o verde (seguro) según el label
REAL del vecindario al que pertenece.

Para cada vecindario elegido, se busca un punto de crimen con imagen
disponible expandiendo el ego-graph adaptativamente (misma lógica que
Paso 4), y se muestra una imagen aleatoria de ese punto.

REQUISITO: correr en la máquina donde viven las imágenes (cluster),
con acceso a graph_output/, embeddings_export/ (metadata.json con paths)
y la carpeta de imágenes real.

Uso:
    python Paso9_qualitative_samples.py --year 2016 --n-per-class 6
"""

import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

RED        = "#dc2626"
GREEN      = "#16a34a"
TEXT_COLOR = "#475569"
SEED       = 42

# Nombres de carpeta de distrito tal como aparecen en image-extraction/
DISTRICT_FOLDER_NAMES = {
    "Inseguros-Barranco-GGZ-2016",
    "Inseguros-La_Victoria-GGZ-2016",
}


def resolve_image_path(stored_path: str, data_path: Path) -> Path:
    """
    Las rutas en metadata.json vienen del entorno donde corrió
    extract_embeddings.py — a veces otra máquina o cluster con una
    estructura de carpetas distinta. Para que esto funcione en CUALQUIER
    máquina, se reconstruye la ruta a partir de --data-path actual,
    tomando solo la parte que empieza en la carpeta de distrito
    (ej. 'Inseguros-Barranco-GGZ-2016/19833096.0/heading_120.jpg') y la
    rejunta con el data_path real de esta máquina.
    """
    parts = Path(stored_path).parts
    for i, part in enumerate(parts):
        if part in DISTRICT_FOLDER_NAMES:
            return data_path / Path(*parts[i:])
    # Fallback: no se encontró el patrón esperado, usar la ruta tal cual
    return Path(stored_path)


# ─── Carga ─────────────────────────────────────────────────────────────────

def rebuild_graph(nodes_csv: Path, edges_csv: Path):
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        G.add_node(int(row["node_id"]), x=row["lon"], y=row["lat"])
    for _, row in edges_df.iterrows():
        G.add_edge(int(row["u"]), int(row["v"]))
    return G, nodes_df


def load_crime_id_to_paths(embeddings_dir: Path, data_path: Path) -> dict:
    with open(embeddings_dir / "metadata.json") as f:
        metadata = json.load(f)
    crime_id_to_paths = {}
    n_unresolved = 0
    for path, cid in zip(metadata["paths"], metadata["crime_ids"]):
        resolved = resolve_image_path(path, data_path)
        if str(resolved) == path and not any(d in path for d in DISTRICT_FOLDER_NAMES):
            n_unresolved += 1
        crime_id_to_paths.setdefault(cid, []).append(str(resolved))

    if n_unresolved > 0:
        print(f"  ⚠ {n_unresolved} rutas no se pudieron reconstruir "
              f"(no se encontró el patrón de carpeta de distrito).")
    sample_resolved = next(iter(crime_id_to_paths.values()))[0]
    print(f"  Ejemplo de ruta resuelta: {sample_resolved}")
    print(f"     ¿Existe en disco? {Path(sample_resolved).exists()}")

    return crime_id_to_paths


def load_crimes_snapped(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["CRIME_ID"]     = df["CRIME_ID"].astype(str)
    df["nearest_node"] = df["nearest_node"].astype(int)
    return df


# ─── Selección de imagen representativa por vecindario ───────────────────────

def find_image_for_node(node, G, crimes_snapped_df, crime_id_to_paths,
                         max_depth=3, rng=None):
    rng = rng or np.random.default_rng(SEED)
    for depth in range(1, max_depth + 1):
        ego_nodes = set(nx.ego_graph(G, node, radius=depth).nodes())
        candidates = crimes_snapped_df[crimes_snapped_df["nearest_node"].isin(ego_nodes)]
        candidates = candidates[candidates["CRIME_ID"].isin(crime_id_to_paths.keys())]
        if len(candidates) > 0:
            chosen_id = rng.choice(candidates["CRIME_ID"].unique())
            return rng.choice(crime_id_to_paths[chosen_id])
    return None


def select_random_nodes_by_label(nodes_df: pd.DataFrame, label_value: str,
                                  n: int, rng: np.random.Generator) -> list:
    candidates = nodes_df[nodes_df["label"] == label_value]["node_id"].tolist()
    if not candidates:
        return []
    n = min(n, len(candidates))
    return list(rng.choice(candidates, size=n, replace=False))


# ─── Plot de grid de imágenes ──────────────────────────────────────────────

def plot_sample_grid(samples: list, output_path: Path, title: str, n_cols: int = 6):
    """samples: lista de tuplas (image_path_or_None, label_str, caption_str)."""
    n = len(samples)
    if n == 0:
        print(f"  ⚠ Sin muestras para '{title}', se omite el plot.")
        return
    n_cols = min(n_cols, n)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.3, n_rows * 2.6), facecolor="white")
    axes = np.atleast_2d(axes)

    for idx in range(n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        ax.set_xticks([]); ax.set_yticks([])

        if idx >= n:
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        path, label, caption = samples[idx]
        if path is not None:
            try:
                img = Image.open(path)
                ax.imshow(img)
            except Exception as e:
                ax.set_facecolor("#fef2f2")
                short_err = str(e)[:40]
                ax.text(0.5, 0.5, f"error:\n{short_err}", ha="center", va="center",
                       transform=ax.transAxes, fontsize=6.5, color=RED, wrap=True)
                print(f"    ⚠ No se pudo abrir {path}: {e}")
        else:
            ax.set_facecolor("#f1f5f9")
            ax.text(0.5, 0.5, "sin imagen\ndisponible", ha="center", va="center",
                   transform=ax.transAxes, fontsize=8, color=TEXT_COLOR)

        border_color = RED if label == "peligroso" else GREEN
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3)
        ax.set_title(caption, fontsize=8, color=TEXT_COLOR)

    fig.suptitle(title, color=TEXT_COLOR, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Guardado: {output_path}")


# ─── Recolección de muestras para un distrito ────────────────────────────────

def collect_samples_for_district(
    distrito: str, year_str: str, graph_dir: Path, embeddings_dir: Path,
    data_path: Path, strat_tag: str, n_per_class: int, rng: np.random.Generator,
):
    base = f"{distrito.replace(' ', '_')}_{year_str}"
    G, nodes_df = rebuild_graph(
        graph_dir / f"{base}_{strat_tag}_nodes.csv",
        graph_dir / f"{base}_edges.csv",
    )
    crimes_snapped_df = load_crimes_snapped(graph_dir / f"{base}_crimes_snapped.csv")
    crime_id_to_paths = load_crime_id_to_paths(embeddings_dir, data_path)

    samples_by_label = {}
    for label_value in ["peligroso", "seguro"]:
        nodes_chosen = select_random_nodes_by_label(nodes_df, label_value, n_per_class, rng)
        samples = []
        for node in nodes_chosen:
            path = find_image_for_node(node, G, crimes_snapped_df, crime_id_to_paths, rng=rng)
            samples.append((path, label_value, f"nodo {node}\n{distrito}"))
        samples_by_label[label_value] = samples

    return samples_by_label


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_pipeline(
    year, graph_dir: Path, embeddings_dir: Path, data_path: Path, output_dir: Path,
    strat_tag: str = "dualgmm", n_per_class: int = 6, seed: int = SEED,
):
    year_str = str(year) if year else "all"
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n  Muestras cualitativas — año {year_str}\n{'='*60}")

    all_samples = {}
    for distrito in ["Barranco", "La Victoria"]:
        print(f"\n  Recolectando muestras de {distrito}…")
        all_samples[distrito] = collect_samples_for_district(
            distrito, year_str, graph_dir, embeddings_dir, data_path,
            strat_tag, n_per_class, rng,
        )
        n_found_pelig = sum(1 for s in all_samples[distrito]["peligroso"] if s[0] is not None)
        n_found_seg   = sum(1 for s in all_samples[distrito]["seguro"] if s[0] is not None)
        print(f"    Peligroso: {n_found_pelig}/{len(all_samples[distrito]['peligroso'])} con imagen")
        print(f"    Seguro:    {n_found_seg}/{len(all_samples[distrito]['seguro'])} con imagen")

    # ── Plot 1: Barranco solo ────────────────────────────────────────────────
    barranco_samples = (all_samples["Barranco"]["peligroso"] +
                        all_samples["Barranco"]["seguro"])
    plot_sample_grid(
        barranco_samples, output_dir / f"Barranco_{year_str}_samples.png",
        f"Muestras cualitativas — Barranco {year_str}\n"
        f"Fila 1: peligroso (borde rojo)  ·  Fila 2: seguro (borde verde)",
        n_cols=n_per_class,
    )

    # ── Plot 2: La Victoria solo ─────────────────────────────────────────────
    victoria_samples = (all_samples["La Victoria"]["peligroso"] +
                        all_samples["La Victoria"]["seguro"])
    plot_sample_grid(
        victoria_samples, output_dir / f"La_Victoria_{year_str}_samples.png",
        f"Muestras cualitativas — La Victoria {year_str}\n"
        f"Fila 1: peligroso (borde rojo)  ·  Fila 2: seguro (borde verde)",
        n_cols=n_per_class,
    )

    # ── Plot 3: Combinado (4 filas: distrito × label) ────────────────────────
    n_combined_per_row = min(n_per_class, 4)   # más compacto para 4 filas
    combined_samples = []
    for distrito in ["Barranco", "La Victoria"]:
        for label_value in ["peligroso", "seguro"]:
            row_samples = all_samples[distrito][label_value][:n_combined_per_row]
            # re-etiquetar caption para incluir distrito+label explícito en la fila
            row_samples = [
                (p, l, f"{distrito}\n{l}") for p, l, _ in row_samples
            ]
            combined_samples.extend(row_samples)
            # rellenar la fila si hay menos muestras de las pedidas (para mantener grid 4xN)
            while len(row_samples) < n_combined_per_row:
                combined_samples.append((None, label_value, f"{distrito}\n{label_value}"))
                row_samples.append(None)

    plot_sample_grid(
        combined_samples, output_dir / f"Combined_{year_str}_samples.png",
        f"Comparación Barranco vs La Victoria — {year_str}\n"
        f"Filas: Barranco-peligroso, Barranco-seguro, La_Victoria-peligroso, La_Victoria-seguro",
        n_cols=n_combined_per_row,
    )

    print(f"\n  ✅ Listo — 3 plots guardados en {output_dir}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paso 9: muestras cualitativas de imágenes")
    parser.add_argument("--year", default="2016")
    parser.add_argument("--graph-dir", default="graph_output")
    parser.add_argument("--embeddings-dir", default="embeddings_export")
    parser.add_argument("--data-path", default="../image-extraction",
                        help="Carpeta raíz con las imágenes reales (default: '../image-extraction')")
    parser.add_argument("--output", default="qualitative_samples")
    parser.add_argument("--strat-tag", default="dualgmm")
    parser.add_argument("--n-per-class", type=int, default=6,
                        help="Cuántos vecindarios aleatorios por clase (peligroso/seguro)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    year = None if args.year == "all" else int(args.year)
    run_pipeline(
        year, Path(args.graph_dir), Path(args.embeddings_dir), Path(args.data_path),
        Path(args.output), strat_tag=args.strat_tag, n_per_class=args.n_per_class, seed=args.seed,
    )


if __name__ == "__main__":
    main()