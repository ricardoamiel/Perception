import os
import pandas as pd


def asegurar_directorios(directorios: list[str]) -> None:
    for d in directorios:
        os.makedirs(d, exist_ok=True)


def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo CSV: {ruta_csv}")
    df = pd.read_csv(ruta_csv, sep=";")
    return df


def guardar_resumen_clusters(df_resumen: pd.DataFrame, ruta_salida: str) -> None:
    df_resumen.to_csv(ruta_salida, index=False)


def guardar_detalle_puntos(df_detalle: pd.DataFrame, ruta_salida: str) -> None:
    df_detalle.to_csv(ruta_salida, index=False)


def crear_directorio_salida_incremental(base_nombre: str = "output") -> str:
    """
    Crea un directorio de salida incremental evitando sobrescritura.
    Si existe 'output', crea 'output_1'; si también existe, 'output_2', etc.
    Devuelve la ruta creada.
    """
    candidato = base_nombre
    contador = 0
    while os.path.exists(candidato):
        contador += 1
        candidato = f"{base_nombre}_{contador}"
    os.makedirs(candidato, exist_ok=True)
    return candidato
