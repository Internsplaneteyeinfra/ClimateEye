#using openmeteo.com
# working code as like original
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta, timezone
import math
import logging
import requests
import openmeteo_requests
from retry_requests import retry


session = requests.Session()
retry_session = retry(session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("AQI-TILES")

app = Flask(__name__)
CORS(app)


def safe_float(x):
    try:
        x = float(x)
        return None if x != x else x
    except:
        return None


def calculate_aqi(pm25, pm10, o3=None, no2=None, so2=None, co=None):
    def calculate_aqi_for_pollutant(concentration, breakpoints):
        if concentration is None:
            return None
        
        for bp in breakpoints:
            c_low, c_high, i_low, i_high = bp
            if c_low <= concentration <= c_high:
                return round(((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low)
        return None

    pm25_bp = [
        (0, 12, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500)
    ]

    pm10_bp = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500)
    ]

    o3_bp = [
        (0, 105.84, 0, 50),
        (107.8, 137.2, 51, 100),
        (139.16, 166.6, 101, 150),
        (168.56, 205.8, 151, 200),
        (207.76, 392, 201, 300)
    ]

    no2_bp = [
        (0, 99.64, 0, 50),
        (101.52, 188, 51, 100),
        (189.88, 676.8, 101, 150),
        (678.68, 1220.12, 151, 200)
    ]

    so2_bp = [
        (0, 91.7, 0, 50),
        (94.32, 196.5, 51, 100),
        (199.12, 484.7, 101, 150),
        (487.32, 796.48, 151, 200)
    ]

    co_bp = [
        (0, 5038, 0, 50),
        (5152.5, 10763, 51, 100),
        (10877.5, 14198, 101, 150),
        (14312.5, 17633, 151, 200)
    ]

    values = [
        calculate_aqi_for_pollutant(pm25, pm25_bp),
        calculate_aqi_for_pollutant(pm10, pm10_bp),
        calculate_aqi_for_pollutant(o3, o3_bp),
        calculate_aqi_for_pollutant(no2, no2_bp),
        calculate_aqi_for_pollutant(so2, so2_bp),
        calculate_aqi_for_pollutant(co, co_bp)
    ]

    values = [v for v in values if v is not None]
    return max(values) if values else None


def fetch_openmeteo_aqi(lat, lon):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide",
            "nitrogen_dioxide", "sulphur_dioxide", "ozone"
        ],
        "timezone": "auto"
    }

    responses = openmeteo.weather_api(url, params=params)
    if not responses:
        return None, None

    response = responses[0]
    hourly = response.Hourly()

    pm10_arr = hourly.Variables(0).ValuesAsNumpy()
    pm25_arr = hourly.Variables(1).ValuesAsNumpy()
    co_arr   = hourly.Variables(2).ValuesAsNumpy()
    no2_arr  = hourly.Variables(3).ValuesAsNumpy()
    so2_arr  = hourly.Variables(4).ValuesAsNumpy()
    o3_arr   = hourly.Variables(5).ValuesAsNumpy()

    start_time = float(hourly.Time())
    interval = float(hourly.Interval()) if hasattr(hourly, "Interval") else 3600
    times = [start_time + i * interval for i in range(len(pm10_arr))]

    now = datetime.now(timezone.utc)
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    current_ts = current_hour.timestamp()

    latest_index = -1
    for i in range(len(times) - 1, -1, -1):
        if times[i] <= current_ts:
            latest_index = i
            break

    if latest_index == -1:
        return None, None

    pm10 = safe_float(pm10_arr[latest_index])
    pm25 = safe_float(pm25_arr[latest_index])
    co   = safe_float(co_arr[latest_index])
    no2  = safe_float(no2_arr[latest_index])
    so2  = safe_float(so2_arr[latest_index])
    o3   = safe_float(o3_arr[latest_index])

    logger.info(
        f"API_DATA | lat={lat:.6f}, lon={lon:.6f} | "
        f"PM2.5={pm25} PM10={pm10} O3={o3} NO2={no2} SO2={so2} CO={co}"
    )

    aqi = calculate_aqi(pm25, pm10, o3, no2, so2, co)
    return aqi, {
        "pm2_5": pm25,
        "pm10": pm10,
        "o3": o3,
        "no2": no2,
        "so2": so2,
        "co": co
    }

def point_in_polygon(lat, lon, polygon):
    inside = False
    x, y = lon, lat

    for i in range(len(polygon)):
        lat1, lon1 = polygon[i]
        lat2, lon2 = polygon[(i + 1) % len(polygon)]
        if ((lat1 > y) != (lat2 > y)) and \
           (x < (lon2 - lon1) * (y - lat1) / (lat2 - lat1) + lon1):
            inside = not inside

    return inside


def generate_tiles_from_polygon(polygon, tile_size_km=2):
    lats = [p[0] for p in polygon]
    lons = [p[1] for p in polygon]

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    lat_step = tile_size_km / 111.0
    tiles = []

    lat = min_lat
    while lat <= max_lat:
        lon_step = tile_size_km / (111.0 * math.cos(math.radians(lat)))
        lon = min_lon
        while lon <= max_lon:
            center_lat = lat + lat_step / 2
            center_lon = lon + lon_step / 2
            if point_in_polygon(center_lat, center_lon, polygon):
                tiles.append({"lat": center_lat, "lon": center_lon})
            lon += lon_step
        lat += lat_step

    return tiles


@app.route("/api/aqi", methods=["POST"])
def get_aqi():
    data = request.get_json()
    lat = data.get("latitude")
    lon = data.get("longitude")

    if not lat or not lon:
        return jsonify({"error": "Latitude and longitude required"}), 400

    aqi, pollutants = fetch_openmeteo_aqi(lat, lon)

    return jsonify({
        "latitude": lat,
        "longitude": lon,
        "aqi": aqi,
        "pollutants": pollutants,
        "hour_used": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:00")
    })


@app.route("/api/aqi/hourly", methods=["POST"])
def get_aqi_hourly():
    data = request.get_json()
    lat = data.get("latitude")
    lon = data.get("longitude")

    if not lat or not lon:
        return jsonify({"error": "Latitude and longitude required"}), 400

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["pm10", "pm2_5", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide"],
        "timezone": "auto"
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    pm10_arr = hourly.Variables(0).ValuesAsNumpy()
    pm25_arr = hourly.Variables(1).ValuesAsNumpy()
    o3_arr   = hourly.Variables(2).ValuesAsNumpy()
    no2_arr  = hourly.Variables(3).ValuesAsNumpy()
    so2_arr  = hourly.Variables(4).ValuesAsNumpy()
    co_arr   = hourly.Variables(5).ValuesAsNumpy()

    result = []
    for i in range(len(pm10_arr)):
        aqi = calculate_aqi(pm25_arr[i], pm10_arr[i], o3_arr[i], no2_arr[i], so2_arr[i], co_arr[i])
        result.append({
            "pm2_5": safe_float(pm25_arr[i]),
            "pm10": safe_float(pm10_arr[i]),
            "o3": safe_float(o3_arr[i]),
            "no2": safe_float(no2_arr[i]),
            "so2": safe_float(so2_arr[i]),
            "co": safe_float(co_arr[i]),
            "aqi": aqi
        })

    logger.info(f"AQI_HOURLY | lat={lat:.6f}, lon={lon:.6f}, result={result}")
    return jsonify(result)


@app.route("/api/aqi/tiles", methods=["POST"])
def get_aqi_tiles():
    data = request.get_json()
    polygon = data.get("polygon")
    tile_size_km = data.get("tile_size_km", 2)

    tiles = generate_tiles_from_polygon(polygon, tile_size_km)

    points = []
    for t in tiles:
        aqi, _ = fetch_openmeteo_aqi(t["lat"], t["lon"])
        logger.info(f"TILE | lat={t['lat']:.6f}, lon={t['lon']:.6f}, AQI={aqi}")
        points.append({
            "lat": t["lat"],
            "lon": t["lon"],
            "aqi": aqi
        })

    return jsonify({
        "hour_used": (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%d %H:00"),
        "points": points
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9100, debug=True)
