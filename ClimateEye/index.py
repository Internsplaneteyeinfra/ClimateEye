from flask import Flask, request, jsonify, send_file

from flask_cors import CORS

import math, random

from io import BytesIO

from PIL import Image, ImageFilter

from cachetools import TTLCache
 
# ==================================================

# App

# ==================================================

app = Flask(__name__)

CORS(app)
 
# ==================================================

# In-memory state + cache

# ==================================================

STATE = {

    "polygon": None,

    "points": []

}
 
TILE_CACHE = TTLCache(maxsize=1500, ttl=900)
 
# ==================================================

# Geometry helpers

# ==================================================

def point_in_polygon(lat, lon, poly):

    inside = False

    j = len(poly) - 1

    for i in range(len(poly)):

        lat_i, lon_i = poly[i]

        lat_j, lon_j = poly[j]

        if ((lon_i > lon) != (lon_j > lon)):

            lat_x = (lon - lon_i) * (lat_j - lat_i) / (lon_j - lon_i) + lat_i

            if lat < lat_x:

                inside = not inside

        j = i

    return inside
 
def tile_intersects(bounds, poly):

    min_lat, min_lon, max_lat, max_lon = bounds

    for lat, lon in poly:

        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:

            return True

    return False
 
# ==================================================

# Tile math

# ==================================================

def pixel_to_latlon(px, py, z, x, y):

    n = 2 ** z

    lon = (x * 256 + px) / (256 * n) * 360 - 180

    lat = math.degrees(

        math.atan(math.sinh(math.pi * (1 - 2 * (y * 256 + py) / (256 * n))))

    )

    return lat, lon
 
def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    dlat = math.radians(lat2 - lat1)

    dlon = math.radians(lon2 - lon1)

    a = (

        math.sin(dlat/2)**2 +

        math.cos(math.radians(lat1)) *

        math.cos(math.radians(lat2)) *

        math.sin(dlon/2)**2

    )

    return 2 * R * math.asin(math.sqrt(a))
 
# ==================================================

# AQI interpolation

# ==================================================

def interpolate(lat, lon, pts, radius=15):

    total = wsum = 0.0

    for p in pts:

        d = haversine(lat, lon, p["lat"], p["lon"])

        if d > radius:

            continue

        d = max(d, 0.001)

        w = 1 / (d ** 1.5)

        total += w * p["aqi"]

        wsum += w

    return int(total / wsum) if wsum else None
 
# ==================================================

# AQI color palette (matches your image)

# ==================================================

def aqi_color(aqi):

    aqi = max(0, min(aqi, 500))

    stops = [0, 50, 100, 150, 200, 300, 500]

    colors = [

        (0,228,0),

        (255,255,0),

        (255,126,0),

        (255,0,0),

        (143,63,151),

        (126,0,35),

        (126,0,35)

    ]

    for i in range(len(stops)-1):

        if stops[i] <= aqi <= stops[i+1]:

            t = (aqi-stops[i])/(stops[i+1]-stops[i])

            return tuple(

                int(colors[i][c] + t*(colors[i+1][c]-colors[i][c]))

                for c in range(3)

            )

    return colors[-1]
 
# ==================================================

# API: Initialize AOI + AQI seeds

# ==================================================

@app.route("/api/aqi/tiles", methods=["POST"])

def init_tiles():

    data = request.get_json()

    poly = data.get("polygon")

    if not poly or len(poly) < 3:

        return jsonify({"error":"Invalid polygon"}), 400
 
    random.seed(hash(str(poly)))
 
    lats = [p[0] for p in poly]

    lons = [p[1] for p in poly]
 
    pts = []

    for _ in range(1800):

        lat = random.uniform(min(lats), max(lats))

        lon = random.uniform(min(lons), max(lons))

        if point_in_polygon(lat, lon, poly):

            pts.append({

                "lat": lat,

                "lon": lon,

                "aqi": random.choice([30, 60, 80, 120, 160, 220, 280])

            })
 
    STATE["polygon"] = poly

    STATE["points"] = pts

    TILE_CACHE.clear()
 
    return jsonify({"status":"ok"})
 
# ==================================================

# API: Raster tile

# ==================================================

@app.route("/api/aqi/raster/<int:z>/<int:x>/<int:y>.png")

def aqi_raster_tile(z, x, y):
 
    cache_key = f"aqi_{z}_{x}_{y}"

    if cache_key in TILE_CACHE:

        return send_file(BytesIO(TILE_CACHE[cache_key]), mimetype="image/png")
 
    TILE_SIZE = 256

    img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))

    pixels = img.load()
 
    polygon = STATE.get("polygon")

    # Reuse the points generated in /api/aqi/tiles (same structure expected by interpolate())
    spatial_points = STATE.get("points", [])
 
    if not polygon or not spatial_points:

        buf = BytesIO()

        img.save(buf, format="PNG")

        buf.seek(0)

        return send_file(buf, mimetype="image/png")
 
    # --------------------------------------------------

    # Tile bounds

    # --------------------------------------------------

    min_lat, min_lon = pixel_to_latlon(0, TILE_SIZE, z, x, y)

    max_lat, max_lon = pixel_to_latlon(TILE_SIZE, 0, z, x, y)
 
    # --------------------------------------------------

    # COARSE GRID (KEY FIX)

    # --------------------------------------------------

    GRID = 32

    step = TILE_SIZE // GRID

    grid = [[None for _ in range(GRID)] for _ in range(GRID)]
 
    # Adaptive radius (same logic as before)

    base_radius = 15.0

    radius_km = base_radius / (2 ** max(0, z - 10))

    radius_km = max(1.0, min(radius_km, 15.0))
 
    for gy in range(GRID):

        for gx in range(GRID):

            px = gx * step

            py = gy * step

            lat, lon = pixel_to_latlon(px, py, z, x, y)
 
            if not point_in_polygon(lat, lon, polygon):

                continue
 
            aqi = interpolate(lat, lon, spatial_points, radius=radius_km)
            grid[gy][gx] = 0 if aqi is None else aqi
 
    # --------------------------------------------------

    # BILINEAR UPSAMPLING (NO PIXELS)

    # --------------------------------------------------

    for py in range(TILE_SIZE):

        for px in range(TILE_SIZE):

            lat, lon = pixel_to_latlon(px, py, z, x, y)
 
            if not point_in_polygon(lat, lon, polygon):

                continue
 
            gx = px / step

            gy = py / step
 
            x0 = int(gx)

            y0 = int(gy)

            x1 = min(x0 + 1, GRID - 1)

            y1 = min(y0 + 1, GRID - 1)
 
            dx = gx - x0

            dy = gy - y0
 
            def g(ix, iy):

                return grid[iy][ix] if grid[iy][ix] is not None else 0
 
            aqi = int(

                g(x0, y0) * (1 - dx) * (1 - dy) +

                g(x1, y0) * dx * (1 - dy) +

                g(x0, y1) * (1 - dx) * dy +

                g(x1, y1) * dx * dy

            )
 
            r, g_col, b = aqi_color(aqi)
 
            alpha = int(min(255, max(120, 100 + (aqi / 300 * 155))))

            pixels[px, py] = (r, g_col, b, alpha)
 
    # --------------------------------------------------

    # FINAL SMOOTHING (IMAGE-LIKE LOOK)

    # --------------------------------------------------

    img = img.filter(ImageFilter.GaussianBlur(radius=1.3))
 
    buf = BytesIO()

    img.save(buf, format="PNG")

    TILE_CACHE[cache_key] = buf.getvalue()

    buf.seek(0)
 
    return send_file(buf, mimetype="image/png")

 
 
# ==================================================

# API: AQI popup value

# ==================================================

@app.route("/api/aqi/value", methods=["POST"])

def aqi_value():

    d = request.get_json()

    lat, lon = d["lat"], d["lon"]

    if not point_in_polygon(lat, lon, STATE["polygon"]):

        return jsonify({"inside":False})

    return jsonify({

        "inside":True,

        "aqi": interpolate(lat, lon, STATE["points"])

    })
 
# ==================================================

@app.route("/api/health")

def health():

    return jsonify({"status":"ok"})
 
# ==================================================

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=8000, debug=True)

 