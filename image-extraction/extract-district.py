import os
import pandas as pd
import math

barranco_bbox = {
    "min_lat": -12.160,
    "max_lat": -12.130,
    "min_lon": -77.030,
    "max_lon": -77.000
}

victoria_bbox = {
    "min_lat": -12.110,
    "max_lat": -12.060,
    "min_lon": -77.030,
    "max_lon": -76.980
}

def generate_points(bbox, district, step_m=80, start_id=0):
    step_lat = step_m / 111000

    avg_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2
    step_lon = step_m / (111000 * math.cos(math.radians(avg_lat)))

    points = []
    cid = start_id

    lat = bbox["min_lat"]
    while lat <= bbox["max_lat"]:
        lon = bbox["min_lon"]
        while lon <= bbox["max_lon"]:
            points.append({
                "cid": cid,
                "lat": round(lat, 7),
                "lng": round(lon, 7),
                "district": district
            })
            cid += 1
            lon += step_lon
        lat += step_lat

    return points, cid

all_points = []
cid = 0

# Barranco
pts, cid = generate_points(barranco_bbox, "Barranco", step_m=80, start_id=cid)
df_barranco = pd.DataFrame(pts)
df_barranco.to_csv("grid_barranco.csv", index=False)

all_points.extend(pts)

# La Victoria
pts, cid = generate_points(victoria_bbox, "La_Victoria", step_m=80, start_id=cid)
df_victoria = pd.DataFrame(pts)
df_victoria.to_csv("grid_la_victoria.csv", index=False)

all_points.extend(pts)

# Dataset combinado
df_all = pd.DataFrame(all_points)
df_all.to_csv("grid_lima_total.csv", index=False)

print("✅ CSVs generados correctamente")