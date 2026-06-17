import os
from src.io_utils import (
    cargar_datos,
    asegurar_directorios,
    guardar_resumen_clusters,
    guardar_detalle_puntos,
    crear_directorio_salida_incremental,
)
from src.pipeline import ejecutar_pipeline


def main():
    # Parámetros iniciales
    ruta_csv = os.environ.get("RUTA_CSV", "crimen_distritos_seleccionados.csv")
    distritos_iniciales = ["La Victoria", "Barranco", "San Isidro"]
    anios = list(range(2016, 2024))

    # Directorio raíz de salida (incremental: output, output_1, output_2, ...)
    base_output_nombre = os.environ.get("OUTPUT_DIR_BASE", "outputs")
    dir_output = crear_directorio_salida_incremental(base_output_nombre)

    # Subcarpetas
    dir_figuras = os.path.join(dir_output, "figures")
    dir_csv = os.path.join(dir_output, "csv")
    asegurar_directorios([dir_figuras, dir_csv])

    # Cargar datos
    df = cargar_datos(ruta_csv)

    # Ejecutar pipeline
    resultados = ejecutar_pipeline(
        df=df,
        distritos=distritos_iniciales,
        anios=anios,
        dir_figuras=dir_figuras,
        min_cluster_size=8,
        min_samples=None,
        # Mapas claros priorizando líneas de calles
        basemap_source="CartoDB.Positron",
        basemap_alpha=0.35,
        overlay_source="Stamen.TonerLines",
        overlay_alpha=0.6,
    )

    # Guardar resultados
    guardar_resumen_clusters(resultados["resumen"], os.path.join(dir_csv, "clusters_resumen.csv"))

    # Guardar detalles por punto (opcional, útil para depuración/inspección)
    for (anio, distrito), detalle in resultados["detalles"].items():
        nombre = f"labels_{anio}_{distrito.replace(' ', '_')}.csv"
        guardar_detalle_puntos(detalle, os.path.join(dir_csv, nombre))


if __name__ == "__main__":
    main()
