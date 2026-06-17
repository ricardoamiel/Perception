"""
extract_embeddings.py
======================
Extrae embeddings (768-dim, normalizados L2) del modelo SimCLR+ViT entrenado,
y los guarda en disco junto con metadata (crime_id, heading, distrito, path).

ESTE SCRIPT REEMPLAZA la sección de extracción de eval.py, pero en vez de
solo usar los embeddings transitoriamente para clustering/UMAP, los persiste
para las siguientes etapas del pipeline (Paso 4: ensamblar matrices por
vecindario).

DIFERENCIAS CLAVE vs eval.py:
  - eval.py NUNCA guarda los embeddings crudos a disco (solo coords UMAP 2D
    + cluster labels). Este script sí los guarda como .npy.
  - El embedding es de 768-dim (salida del backbone ViT, h), NO 512.
    El 512 solo existe dentro del projector de SimCLR (descartado en
    inferencia, como ya hace eval.py correctamente).
  - Captura crime_id y heading por imagen (eval.py solo captura distrito).

CONVENCIÓN DE CARPETAS CONFIRMADA:
  data/Inseguros-Barranco-GGZ-2016/19833096.0/heading_120.jpg
  El crime_id vive en el nombre de la SUBCARPETA, con sufijo '.0' siempre
  (probablemente de un cast float→str del scraping original). Se normaliza
  con normalize_crime_id() — la misma función usada en graph_pipeline.py
  para el CSV de crímenes, así ambos lados del join quedan en el mismo
  formato sin importar si el ID original era int, float o string.

BLACKLIST DE IMÁGENES INVÁLIDAS:
  filter_invalid_images.py genera un JSON {ruta_relativa: razón} con
  imágenes corruptas/inválidas (ej. archivos truncados de pocos KB) que NO
  deben usarse para extraer embeddings. Este script lo carga con
  --blacklist y excluye esas imágenes ANTES de correr el modelo.
  Las rutas del JSON son relativas a DATA_PATH (ej.
  "Inseguros-Barranco-GGZ-2016/19833096.0/heading_120.jpg").

Uso:
    # Corriendo desde research_pt2/graph/ (estructura real: image-extraction/
    # es carpeta hermana, sin subcarpeta 'data' intermedia)
    python extract_embeddings.py \\
        --ckpt ../checkpoints_simclr_vit/simclr_perception_224x_224_best_epoch_50.pt \\
        --data-path ../image-extraction \\
        --blacklist ../gt/invalid_images.json \\
        --output embeddings_export
"""

import os
import re
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
import numpy as np

# ─── Config ──────────────────────────────────────────────────────────────────
# DATA_PATH es configurable vía --data-path (ver abajo). El default asume que
# se corre desde research_pt2/graph/ y que las imágenes viven directamente en
# research_pt2/image-extraction/ (SIN subcarpeta 'data' intermedia):
#
#   research_pt2/
#     distritos/          (CSV maestro de crímenes)
#     graph/               ← se corre el script desde aquí
#     gt/                  (filter_invalid_images.py + blacklist)
#     image-extraction/    (las imágenes Street View, carpetas por distrito)

DEFAULT_DATA_PATH = "../image-extraction"
IMG_SIZE   = 224
BATCH_SIZE = 128
EMBED_DIM  = 768   # salida del ViT-base backbone (CLS token), NO 512
DEFAULT_OUTPUT_DIR = "embeddings_export"

# Nombres de las carpetas de distrito tal como aparecen en `data/`
DISTRICT_FOLDER_NAMES = {
    "Inseguros-Barranco-GGZ-2016",
    "Inseguros-La_Victoria-GGZ-2016",
}


# ─── Modelo (idéntico a eval.py, solo retorna backbone features) ─────────────

class SimCLR_Model(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model
        self.projector = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 128)
        )

    def forward(self, x):
        h = self.encoder(x)
        return h   # backbone feature (768-dim) — estándar SimCLR para downstream


# ─── Parseo de crime_id y heading desde el path ───────────────────────────────

def normalize_crime_id(raw) -> str:
    """
    Canonicaliza un crime_id al mismo formato usado en graph_pipeline.py,
    sin importar si llega como int, float, o string con/sin '.0'.
    CRÍTICO: si esto no coincide exactamente con el formato del CSV de
    crímenes, el join en Paso 4 falla silenciosamente (0% overlap).
    """
    try:
        val = float(raw)
        if val == int(val):
            return f"{int(val)}.0"
        return str(val)
    except (ValueError, TypeError):
        return str(raw)


def parse_crime_id_and_heading(path: str) -> tuple[str, int]:
    """
    Extrae (crime_id, heading_deg) de la ruta de la imagen.

    Convención CONFIRMADA:
      data/Inseguros-Barranco-GGZ-2016/19833096.0/heading_120.jpg
      → crime_id = nombre de la subcarpeta (normalizado con '.0')
      → heading  = dígitos después de "heading_"
    """
    parent_folder = os.path.basename(os.path.dirname(path))
    filename      = os.path.basename(path)

    heading_match = re.search(r"heading_(\d+)", filename)
    heading = int(heading_match.group(1)) if heading_match else -1

    if parent_folder not in DISTRICT_FOLDER_NAMES and parent_folder != "":
        crime_id = normalize_crime_id(parent_folder)
    else:
        # Fallback por si la estructura fuera plana (no es el caso confirmado,
        # pero se deja por robustez)
        id_match = re.search(r"([\d.]+)[_\-]?heading_", filename)
        if id_match:
            crime_id = normalize_crime_id(id_match.group(1))
        else:
            crime_id = filename.split("_heading_")[0] if "_heading_" in filename \
                       else filename.split("heading_")[0]

    return crime_id, heading


def load_blacklist(blacklist_path: str | None) -> dict:
    """
    Carga el JSON de filter_invalid_images.py: {ruta_relativa: razón}.
    Las rutas son relativas a DATA_PATH, ej:
      "Inseguros-Barranco-GGZ-2016/19833096.0/heading_120.jpg"
    """
    if not blacklist_path:
        print("  ℹ No se especificó --blacklist. Se usarán TODAS las imágenes "
              "(incluyendo posibles archivos corruptos).")
        return {}

    if not os.path.exists(blacklist_path):
        raise FileNotFoundError(f"No se encontró el blacklist en: {blacklist_path}")

    with open(blacklist_path) as f:
        blacklist = json.load(f)

    print(f"  Blacklist cargado: {len(blacklist):,} imágenes a excluir")

    # Resumen de razones (útil para verificar que el filtro tiene sentido)
    reasons = {}
    for reason in blacklist.values():
        key = reason.split(" (")[0]   # "tiny_file (10550 bytes)" → "tiny_file"
        reasons[key] = reasons.get(key, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:<20}: {count:,}")

    return blacklist


# ─── Dataset ───────────────────────────────────────────────────────────────────

class InferenceDataset(Dataset):
    def __init__(self, root_dir, transform, blacklist: dict | None = None):
        blacklist = blacklist or {}
        self.paths, self.districts, self.crime_ids, self.headings = [], [], [], []
        n_skipped = 0

        for root, _, files in os.walk(root_dir):
            for f in files:
                if "heading_" in f and f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, f)

                    # Ruta relativa a root_dir, igual al formato del JSON
                    # de filter_invalid_images.py (usa '/' siempre, no os.sep,
                    # por si se corre en un sistema distinto al de generación)
                    rel_path = os.path.relpath(full_path, root_dir).replace(os.sep, "/")
                    if rel_path in blacklist:
                        n_skipped += 1
                        continue

                    crime_id, heading = parse_crime_id_and_heading(full_path)
                    self.paths.append(full_path)
                    self.districts.append("Barranco" if "Barranco" in root else "La Victoria")
                    self.crime_ids.append(crime_id)
                    self.headings.append(heading)

        self.transform = transform
        self.n_skipped = n_skipped

        if blacklist:
            print(f"  Imágenes excluidas por blacklist: {n_skipped:,} / "
                  f"{n_skipped + len(self.paths):,} encontradas")
            if n_skipped != len(blacklist):
                print(f"  ⚠ El blacklist tiene {len(blacklist):,} entradas pero solo se "
                      f"excluyeron {n_skipped:,} — revisa que las rutas relativas coincidan "
                      f"(case-sensitivity, separadores, etc.)")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), idx   # idx para reconstruir metadata después


def sanity_check_parsing(dataset: InferenceDataset, data_path: str, n_samples: int = 10):
    print("\n  🔍 Verificación de parseo (primeros 10 ejemplos):")
    print(f"  {'path (truncado)':<55} {'crime_id':<14} {'heading':<8}")
    for i in range(min(n_samples, len(dataset))):
        p = dataset.paths[i]
        short_p = "…" + p[-50:] if len(p) > 50 else p
        print(f"  {short_p:<55} {dataset.crime_ids[i]:<14} {dataset.headings[i]:<8}")

    n_unique_ids    = len(set(dataset.crime_ids))
    n_total_images  = len(dataset)
    avg_imgs_per_id = n_total_images / max(n_unique_ids, 1)

    print(f"\n  Total imágenes        : {n_total_images:,}")
    print(f"  IDs únicos detectados  : {n_unique_ids:,}")
    print(f"  Promedio imágenes/ID   : {avg_imgs_per_id:.1f}  (esperado: ~12, "
          f"menos si el blacklist excluyó imágenes)")

    if avg_imgs_per_id < 1.5 or avg_imgs_per_id > 30:
        print(f"\n  ⚠️⚠️ ADVERTENCIA: el promedio de imágenes/ID está lejos de 12.")
        print(f"     El parseo de crime_id probablemente está mal.")
        print(f"     Corre: ls -la {data_path}/Inseguros-Barranco-GGZ-2016 | head -20")
        print(f"     y ajusta parse_crime_id_and_heading() con la convención real.")
        print(f"     (El script continuará, pero revisa esto antes de usar los resultados)\n")


# ─── Extracción ────────────────────────────────────────────────────────────────

def run_extraction(
    ckpt_path: str,
    output_dir: str,
    data_path: str,
    blacklist_path: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Extrayendo embeddings en {device}…")
    print(f"  DATA_PATH: {os.path.abspath(data_path)}")

    blacklist = load_blacklist(blacklist_path)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = InferenceDataset(data_path, transform, blacklist=blacklist)
    print(f"  {len(dataset):,} imágenes válidas encontradas con patrón 'heading_'.")
    if len(dataset) == 0:
        print(f"  ❌ No se encontraron imágenes. Revisa --data-path "
              f"({os.path.abspath(data_path)}).")
        return
    sanity_check_parsing(dataset, data_path)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    base  = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    model = SimCLR_Model(base).to(device)

    print(f"\n  Cargando checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_embeddings = np.zeros((len(dataset), EMBED_DIM), dtype=np.float32)

    with torch.no_grad():
        for imgs, idxs in tqdm(loader, desc="Extrayendo features"):
            imgs  = imgs.to(device)
            feats = model(imgs)
            feats = F.normalize(feats, dim=-1)   # L2-normalizado, igual que eval.py
            all_embeddings[idxs.numpy()] = feats.cpu().numpy()

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "embeddings.npy"), all_embeddings)
    print(f"\n  ✅ Guardado: embeddings.npy  shape={all_embeddings.shape}")

    metadata = {
        "paths":            dataset.paths,
        "districts":        dataset.districts,
        "crime_ids":        dataset.crime_ids,
        "headings":         dataset.headings,
        "n_excluded_blacklist": dataset.n_skipped,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    print(f"  ✅ Guardado: metadata.json  ({len(dataset.paths):,} entradas)")

    # Reporte rápido por distrito
    districts_arr = np.array(dataset.districts)
    for d in set(dataset.districts):
        n_d        = (districts_arr == d).sum()
        n_ids_d    = len(set(np.array(dataset.crime_ids)[districts_arr == d]))
        print(f"    {d:<15}: {n_d:,} imágenes, {n_ids_d:,} crime_ids únicos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraer y persistir embeddings ViT+SimCLR")
    parser.add_argument("--ckpt", required=True,
                        help="Path al checkpoint .pt (ej. ../checkpoints_simclr_vit/simclr_perception_224x_224_best_epoch_50.pt)")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                        help=f"Carpeta raíz con las imágenes (default: '{DEFAULT_DATA_PATH}', "
                             f"asumiendo que se corre desde research_pt2/graph/)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--blacklist", default=None,
                        help="Path al JSON de filter_invalid_images.py "
                             "(ej. ../gt/invalid_images.json). Si se omite, se usan todas las imágenes.")
    args = parser.parse_args()
    run_extraction(args.ckpt, args.output, args.data_path, blacklist_path=args.blacklist)