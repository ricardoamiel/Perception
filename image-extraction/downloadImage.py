import os
import requests
from geopy.distance import geodesic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MAPILLARY_ACCESS_TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN")
GOOGLE_STREETVIEW_API_KEY = os.getenv("GOOGLE_STREETVIEW_API_KEY")


def get_images_mapillary(lat:float, lon:float,OUTPUT_DIR)-> None:
    ACCESS_TOKEN = MAPILLARY_ACCESS_TOKEN

    if not ACCESS_TOKEN:
        raise ValueError("MAPILLARY_ACCESS_TOKEN no encontrado en .env")
    #OUTPUT_DIR      = "imagenes_rotacion"
    MAX_IMAGES      = 30  

    punto_objetivo = (lat, lon)

    dlat = 20 / 111000  #
    dlon = 20 / 108000  #


    min_lat = lat - dlat
    max_lat = lat + dlat
    min_lon = lon - dlon
    max_lon = lon + dlon
    bbox = [min_lon, min_lat, max_lon, max_lat]


    # Crear carpeta si no existe
    #os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------
    # Llamada a la API de Mapillary
    # -------------------------
    url = "https://graph.mapillary.com/images"
    params = {
        "access_token": ACCESS_TOKEN,
        "fields": "id,thumb_2048_url,computed_geometry,compass_angle,captured_at",
        "bbox": ",".join(map(str, bbox)),
        "limit": MAX_IMAGES
    }

    response = requests.get(url, params=params)
    print("Código de respuesta:", response.status_code)

    if response.status_code != 200:
        print("Error al conectarse con la API.")
        exit()

    data = response.json()
    imagenes = data.get("data", [])
    print(f"Total imágenes encontradas en la bbox: {len(imagenes)}")

    # -------------------------
    # Calcular distancia al punto central y ordenar
    # -------------------------
    imagenes_info = []
    for img in imagenes:
        coords = img.get("computed_geometry", {}).get("coordinates", [])
        if len(coords) == 2:
            punto_img = (coords[1], coords[0])  # (lat, lon)
            dist = geodesic(punto_objetivo, punto_img).meters
            imagenes_info.append((dist, img))

    imagenes_info = [
        (dist, img) for dist, img in imagenes_info if dist < 20  # Solo las dentro de 15 m
    ]
    # Ordenar por distancia y tomar las 8 más cercanas
    imagenes_info.sort(key=lambda x: x[0])
    imagenes_seleccionadas = imagenes_info[:8]

    # -------------------------
    # Descargar las imágenes
    # -------------------------
    for dist, img in imagenes_seleccionadas:
        img_url = img.get("thumb_2048_url")
        heading = img.get("compass_angle", 0)
        if img_url:
            print(f"Descargando imagen a {int(dist)} m, rumbo {int(heading)}°")
            try:
                img_data = requests.get(img_url).content
                filename = os.path.join(OUTPUT_DIR, f"{img['id']}_r{int(heading)}.jpg")
                with open(filename, "wb") as f:
                    f.write(img_data)
            except Exception as e:
                print(f"Error al descargar {img_url}: {e}")

    print("✅ Descarga finalizada.")

def get_images_google(lat:float, lon:float,output_folder)-> None:
    # Parámetros
    API_KEY = GOOGLE_STREETVIEW_API_KEY
    if not API_KEY:
        raise ValueError("GOOGLE_STREETVIEW_API_KEY no encontrado en .env")
    latitude = lat
    longitude = lon
    fov = 90
    pitch = 0
    size = "640x400"

    # Crear carpeta para imágenes
    #output_folder = "streetview_360"
    #os.makedirs(output_folder, exist_ok=True)

    # Capturar imágenes con rotaciones de 45°
    for heading in range(0, 360, 30):
        url = (
            f"https://maps.googleapis.com/maps/api/streetview"
            f"?size={size}&location={latitude},{longitude}"
            f"&heading={heading}&pitch={pitch}&fov={fov}&key={API_KEY}"
        )
        response = requests.get(url)

        if response.status_code == 200:
            filename = f"{output_folder}/heading_{heading}.jpg"
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"Imagen guardada: {filename}")
        else:
            print(f"Error al obtener imagen para heading {heading}: {response.status_code}")



#lat = -23.593125119574
#lon = -46.658361379384
#output_folder = "streetview_360"
import os
year = 2016
distrito = "La_Victoria"
selected_points = pd.read_csv(f'outputs/csv/labels_{year}_{distrito}.csv')

for i, row in selected_points.iterrows():
    if i>=0:
        id = row['ID']
        lat = row['latitude']
        lng = row['longitude']
        odir = f'Inseguros-{distrito}-GGZ-{year}/'+str(id)

        # --- LÓGICA DE VERIFICACIÓN ---
        # Si la carpeta existe Y tiene archivos dentro, asumimos que ya se procesó
        if os.path.exists(odir) and len(os.listdir(odir)) > 0:
            print(f"--- Saltando punto {i} (ID: {id}) - Ya descargado.")
            continue
        # ------------------------------

        os.makedirs(odir, exist_ok=True)
        #get_images_mapillary(lat, lng,odir)
        #get_images_google(lat, lng,odir)
        #print(f" estoy sacando de {i} y faltan {selected_points.shape[0]}")
        try:
            get_images_mapillary(lat, lng, odir)
            get_images_google(lat, lng, odir)
            print(f" estoy sacando de {i} y faltan {selected_points.shape[0] - i - 1}")
        except Exception as e:
            print(f"!!!! Error en punto {id}: {e}")
print("tERMINÉ")

#get_images_mapillary(lat, lon)
#get_images_google(lat, lon)