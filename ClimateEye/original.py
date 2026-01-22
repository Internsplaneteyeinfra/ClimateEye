# the heatmap generation code with improved tile generation and rasterization
# original code is different 

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from datetime import datetime, timedelta, timezone
import requests
import requests_cache
from retry_requests import retry
import logging,os
from groq import Groq
from dotenv import load_dotenv
import math
import random
from io import BytesIO
from PIL import Image
from cachetools import TTLCache
import numpy as np

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")


AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Moderate"),
    (101, 150, "Poor"),
    (151, 200, "Unhealthy"),
    (201, 300, "Severe"),
    (301, float("inf"), "Hazardous"),
]

def get_aqi_category(aqi):
    for low, high, label in AQI_CATEGORIES:
        if low <= aqi <= high:
            return label
    return "Unknown"

def get_aqi_color(aqi):
    """
    Get color for AQI value based on US EPA standards
    Returns hex color code and category
    """
    if aqi is None:
        return {"color": "#808080", "category": "Unknown", "label": "No Data"}
    
    if aqi <= 50:
        return {"color": "#00E400", "category": "Good", "label": "Good"}
    elif aqi <= 100:
        return {"color": "#FFFF00", "category": "Moderate", "label": "Moderate"}
    elif aqi <= 150:
        return {"color": "#FF7E00", "category": "Poor", "label": "Unhealthy for Sensitive Groups"}
    elif aqi <= 200:
        return {"color": "#FF0000", "category": "Unhealthy", "label": "Unhealthy"}
    elif aqi <= 300:
        return {"color": "#8F3F97", "category": "Severe", "label": "Very Unhealthy"}
    else:
        return {"color": "#7E0023", "category": "Hazardous", "label": "Hazardous"}

def point_in_polygon(lat, lon, polygon):
    """
    Check if a point (lat, lon) is inside a polygon using ray casting algorithm
    
    Parameters:
    - lat, lon: Point coordinates (latitude, longitude)
    - polygon: List of [latitude, longitude] pairs forming a closed polygon
    
    Returns:
    - True if point is inside polygon, False otherwise
    """
    if not polygon or len(polygon) < 3:
        return False
    
    # Ensure polygon is closed (first point == last point)
    poly = polygon[:]
    if poly[0] != poly[-1]:
        poly.append(poly[0])
    
    inside = False
    j = len(poly) - 1
    
    for i in range(len(poly) - 1):
        xi, yi = poly[i]  # xi = lat, yi = lon
        xj, yj = poly[j]  # xj = lat, yj = lon
        
      
        if ((yi > lon) != (yj > lon)):  
            if yj != yi: 
                x_intersect = (lon - yi) * (xj - xi) / (yj - yi) + xi
                if lat < x_intersect: 
                    inside = not inside  
        j = i
    return inside

def get_polygon_bounds(polygon):
    """
    Calculate bounding box from polygon coordinates
    
    Parameters:
    - polygon: List of [latitude, longitude] pairs
    
    Returns:
    - Dictionary with min_lat, max_lat, min_lon, max_lon
    """

    if not polygon:
        return None
    
    lats = [point[0] for point in polygon]
    lons = [point[1] for point in polygon]
    
    return {
        "min_lat": min(lats),
        "max_lat": max(lats),
        "min_lon": min(lons),
        "max_lon": max(lons)
    }

def generate_tiles_from_bounds(min_lat, max_lat, min_lon, max_lon, tile_size_km=0.5):
    """
    Generate uniform grid tiles within a bounding box using pure coordinate-based geometry.
    
    This function is completely direction-agnostic. It treats coordinates purely as spatial values
    without any directional bias. All calculations are based on coordinate mathematics only.
    
    Algorithm:
    1. Convert tile size from kilometers to coordinate degrees (accounting for latitude-dependent longitude scaling)
    2. Calculate grid dimensions based on spatial coordinate ranges
    3. Generate uniform grid cells covering the bounding box
    4. Each tile is defined by its coordinate boundaries and center point
    
    Parameters:
    - min_lat, max_lat: Minimum and maximum latitude coordinates (degrees)
    - min_lon, max_lon: Minimum and maximum longitude coordinates (degrees)
    - tile_size_km: Size of each tile in kilometers (default 0.5 km = 500m)
    
    Returns:
    - List of tile dictionaries, each containing:
      - tile_id: Unique identifier (row_column format)
      - center: {latitude, longitude} - center point for AQI calculation
      - bounds: {min_lat, max_lat, min_lon, max_lon} - coordinate boundaries (direction-agnostic)
    """

    
    # Validate bounds (purely geometric validation)
    if max_lat <= min_lat or max_lon <= min_lon:
        return []

    # 1 degree latitude ≈ 111 km (constant, independent of location)
    # 1 degree longitude ≈ 111 km * cos(latitude) (varies by latitude due to Earth's curvature)
    avg_lat = (min_lat + max_lat) / 2
    lat_km_per_deg = 111.0
    lon_km_per_deg = 111.0 * math.cos(math.radians(avg_lat))
    
    # Convert tile size from kilometers to coordinate degrees
    # This ensures each tile represents approximately tile_size_km × tile_size_km area
    tile_lat_deg = tile_size_km / lat_km_per_deg
    tile_lon_deg = tile_size_km / lon_km_per_deg
    
    # Calculate spatial ranges (coordinate differences)
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    
    # Calculate number of tiles needed to cover the bounding box
    # Use ceiling to ensure full coverage
    num_tiles_lat = max(1, int(math.ceil(lat_range / tile_lat_deg)))
    num_tiles_lon = max(1, int(math.ceil(lon_range / tile_lon_deg)))
    
    tiles = []
    
    # Generate uniform grid - iterate through grid positions (i, j)
    for i in range(num_tiles_lat):
        for j in range(num_tiles_lon):
            # Calculate tile coordinate boundaries (purely geometric)
            tile_min_lat = min_lat + (i * tile_lat_deg)
            tile_max_lat = min(max_lat, tile_min_lat + tile_lat_deg)
            tile_min_lon = min_lon + (j * tile_lon_deg)
            tile_max_lon = min(max_lon, tile_min_lon + tile_lon_deg)
            
            # Calculate center point (for AQI data fetching)
            center_lat = (tile_max_lat + tile_min_lat) / 2
            center_lon = (tile_max_lon + tile_min_lon) / 2
            
            tiles.append({
                "tile_id": f"{i}_{j}",
                "center": {
                    "latitude": center_lat,
                    "longitude": center_lon
                },
                "bounds": {
                    "min_lat": tile_min_lat,
                    "max_lat": tile_max_lat,
                    "min_lon": tile_min_lon,
                    "max_lon": tile_max_lon
                }
            })
    
    return tiles

def generate_tiles_for_polygon(polygon, tile_size_km=0.5):
    """
    Generate grid tiles within a polygon geometry using coordinate-based calculations.
    
    Algorithm:
    1. Calculate bounding box from polygon coordinates (for grid generation)
    2. Generate uniform grid of tiles covering the bounding box
    3. Filter tiles to only include those whose centers are inside the polygon
    
    All calculations are coordinate-based and direction-agnostic.
    
    Parameters:
    - polygon: List of [latitude, longitude] pairs forming a closed polygon
    - tile_size_km: Size of each tile in kilometers (default 0.5 km = 500m)
    
    Returns:
    - List of tile dictionaries with center coordinates and bounds
    """
    if not polygon or len(polygon) < 3:
        return []
    
    # Get bounding box from polygon (coordinate-based)
    bounds = get_polygon_bounds(polygon)
    if not bounds:
        return []
    
    min_lat = bounds["min_lat"]
    max_lat = bounds["max_lat"]
    min_lon = bounds["min_lon"]
    max_lon = bounds["max_lon"]
    
    # Generate grid tiles covering the bounding box (coordinate-based)
    all_tiles = generate_tiles_from_bounds(min_lat, max_lat, min_lon, max_lon, tile_size_km)
    
    # Filter tiles to only include those whose centers are inside the polygon
    filtered_tiles = []
    for tile in all_tiles:
        center_lat = tile["center"]["latitude"]
        center_lon = tile["center"]["longitude"]
        
        # Only include tiles whose centers are inside the polygon
        if point_in_polygon(center_lat, center_lon, polygon):
            filtered_tiles.append(tile)
    
    return filtered_tiles

# NEW: AQI visualization palette (GLOBAL)
AQI_VIS_PARAMS = {
    "min": 0,
    "max": 500,
    "palette": [
        "#00e400",  # Good
        "#ffff00",  # Moderate
        "#ff7e00",  # Poor
        "#ff0000",  # Unhealthy
        "#8f3f97",  # Very Unhealthy
        "#7e0023",  # Hazardous
    ]
}

def latlon_to_tile(lat, lon, z):
    """Convert lat/lon to tile coordinates ."""
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def get_palette_color(aqi):
    """
    Smooth gradient color for raster (SAFE)
    """
    aqi = max(0, min(aqi, 300))  

    stops = [0, 50, 100, 150, 200, 300]
    colors = [
        (0, 228, 0),     # Green
        (255, 255, 0),   # Yellow
        (255, 126, 0),   # Orange
        (255, 0, 0),     # Red
        (143, 63, 151),  # Purple
        (126, 0, 35)     # Maroon
    ]

    for i in range(len(stops) - 1):
        if stops[i] <= aqi <= stops[i + 1]:
            t = (aqi - stops[i]) / (stops[i + 1] - stops[i])
            return tuple(
                int(colors[i][j] + t * (colors[i + 1][j] - colors[i][j]))
                for j in range(3)
            )

    return colors[-1]


def interpolate_aqi(lat, lon, spatial_points, radius_km=15.0):
    """
    Inverse Distance Weighting (IDW)
    Smooth AQI interpolation
    """
    total = 0.0
    weight_sum = 0.0

    for p in spatial_points:
        tlat = p["lat"]
        tlon = p["lon"]
        aqi = p["aqi"]
      
        if aqi is None:
            continue

        d = haversine_km(lat, lon, tlat, tlon)
        if d > radius_km:
            continue

        d = max(d, 0.0001) 

        w = 1 / (d ** 2)
        total += w * aqi
        weight_sum += w

    return int(total / weight_sum) if weight_sum else None


def generate_spatial_points_from_tile(tile, points_per_tile=6):
    """
    Generate spatial points from AQI grid tile (VIEW LAYER ONLY)
    """
    spatial_points = []

    b = tile["bounds"]
    aqi = tile["aqi"]

    # if aqi is None:
    #     return points

    for _ in range(points_per_tile):
        lat = random.uniform(b["min_lat"], b["max_lat"])
        lon = random.uniform(b["min_lon"], b["max_lon"])

        spatial_points.append({
            "lat": lat,
            "lon": lon,
            "aqi": aqi
        })

    return spatial_points


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2 +
        math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))

def pixel_to_latlon(px, py, z, x, y):
    n = 2 ** z
    lon = (x * 256 + px) / (256 * n) * 360.0 - 180.0
    lat_rad = math.atan(
        math.sinh(math.pi * (1 - 2 * (y * 256 + py) / (256 * n)))
    )
    lat = math.degrees(lat_rad)
    return lat, lon


AQI_TILE_CACHE = {
    "tiles": [],
    "spatial_points": [],
    "meta": {}
}

# Tile-level cache for raster tiles 
TILE_CACHE = TTLCache(maxsize=5000, ttl=3600)

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/api/aqi/tiles', methods=['POST'])
def get_aqi_tiles():
    data = request.get_json()

    polygon = data.get("polygon")
    date = data.get("date")
    tile_size_km = data.get("tile_size_km", 0.5)

    if not polygon or len(polygon) < 3:
        return jsonify({"error": "Valid polygon required"}), 400
    if not date:
        return jsonify({"error": "date is required"}), 400

    # 1️⃣ GRID GENERATION
    tiles = generate_tiles_for_polygon(polygon, tile_size_km)
    bounds = get_polygon_bounds(polygon)

    if not tiles:
        return jsonify({"error": "No tiles generated for polygon"}), 400

    tile_results = []
    aqi_values = []

    # Dummy AQI (replace later with real AQI)
    DUMMY_AQI = [25, 75, 110, 155, 190, 260, 310]

    for i, tile in enumerate(tiles):
        aqi = DUMMY_AQI[i % len(DUMMY_AQI)]
        color_info = get_aqi_color(aqi)

        tile_results.append({
            "tile_id": tile["tile_id"],
            "center": tile["center"],
            "bounds": tile["bounds"],
            "aqi": aqi,
            "color": color_info["color"],
            "category": color_info["category"],
            "label": color_info["label"]
        })

        aqi_values.append(aqi)

    # 2️⃣ CACHE (IMPORTANT)
    AQI_TILE_CACHE.clear()
    AQI_TILE_CACHE["tiles"] = tile_results
    AQI_TILE_CACHE["polygon"] = polygon
    AQI_TILE_CACHE["bounds"] = bounds

    # 3️⃣ SPATIAL POINTS (VIEW ONLY) - Increase density for better interpolation
    spatial_points = []
    for tile in tile_results:
        spatial_points.extend(
            generate_spatial_points_from_tile(tile, points_per_tile=5)  # Increased from 25 to 50
        )

    AQI_TILE_CACHE["spatial_points"] = spatial_points
    
    # Clear tile cache to force regeneration
    TILE_CACHE.clear()
    
    logger.info(f"Generated {len(tile_results)} tiles with {len(spatial_points)} spatial points for polygon")

    return jsonify({
        "tiles": tile_results,
        "bounds": bounds,
        "aqi_tile_url": "http://localhost:8000/api/aqi/raster/{z}/{x}/{y}.png"
    }), 200



@app.route("/api/aqi/raster/<int:z>/<int:x>/<int:y>.png")
def aqi_raster_tile(z, x, y):

    cache_key = f"aqi_{z}_{x}_{y}"
    if cache_key in TILE_CACHE:
        return send_file(BytesIO(TILE_CACHE[cache_key]), mimetype="image/png")

    TILE_SIZE = 256
    img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE))
    pixels = img.load()

    polygon = AQI_TILE_CACHE.get("polygon")
    spatial_points = AQI_TILE_CACHE.get("spatial_points", [])

    if not polygon or not spatial_points:
        # Return transparent tile instead of empty image
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    # Calculate tile bounds with buffer for better point matching
    min_lat, min_lon = pixel_to_latlon(0, TILE_SIZE, z, x, y)
    max_lat, max_lon = pixel_to_latlon(TILE_SIZE, 0, z, x, y)
    
    # Increase buffer based on zoom level (higher zoom = smaller buffer needed)
    buffer = max(0.01, 0.1 / (2 ** (z - 10)))  # Adaptive buffer

    relevant_points = [
        p for p in spatial_points
        if min_lat - buffer <= p["lat"] <= max_lat + buffer
        and min_lon - buffer <= p["lon"] <= max_lon + buffer
    ]

    # If no relevant points, try using all spatial points (fallback)
    if not relevant_points:
        relevant_points = spatial_points

    # Calculate adaptive radius based on zoom level
    # Higher zoom = smaller radius needed
    base_radius = 15.0
    radius_km = base_radius / (2 ** max(0, z - 10))
    radius_km = max(1.0, min(radius_km, 15.0))  # Clamp between 1 and 15 km

    pixels_rendered = 0
    for py in range(TILE_SIZE):
        for px in range(TILE_SIZE):
            lat, lon = pixel_to_latlon(px, py, z, x, y)

            if not point_in_polygon(lat, lon, polygon):
                pixels[px, py] = (0, 0, 0, 0)
                continue

            aqi = interpolate_aqi(lat, lon, relevant_points, radius_km=radius_km)
            if aqi is None:
                pixels[px, py] = (0, 0, 0, 0)
                continue

            r, g, b = get_palette_color(aqi)
            # Increase alpha for better visibility (150-255 range)
            alpha = int(min(255, max(150, 100 + (aqi / 300 * 155))))
            pixels[px, py] = (r, g, b, alpha)
            pixels_rendered += 1

    # Log for debugging
    if pixels_rendered > 0:
        logger.debug(f"Tile {z}/{x}/{y}: Rendered {pixels_rendered} pixels, {len(relevant_points)} points, radius={radius_km:.2f}km")

    buf = BytesIO()
    img.save(buf, format="PNG")
    TILE_CACHE[cache_key] = buf.getvalue()
    buf.seek(0)

    return send_file(buf, mimetype="image/png")



@app.route('/api/aqi/value', methods=['POST'])
def get_aqi_value():
    """
    Get AQI value for a specific lat/lon point
    Used for click popups on the map
    """
    try:
        data = request.get_json()
        lat = data.get('lat')
        lon = data.get('lon')
        
        if lat is None or lon is None:
            return jsonify({'error': 'Latitude and longitude are required'}), 400
        
        polygon = AQI_TILE_CACHE.get("polygon")
        spatial_points = AQI_TILE_CACHE.get("spatial_points", [])
        
        # Check if point is inside polygon
        inside = False
        if polygon:
            inside = point_in_polygon(lat, lon, polygon)
        
        if not inside:
            return jsonify({
                'inside': False,
                'aqi': None,
                'category': None
            }), 200
        
        # Interpolate AQI for this point
        aqi = interpolate_aqi(lat, lon, spatial_points, radius_km=15.0)
        
        if aqi is None:
            return jsonify({
                'inside': True,
                'aqi': None,
                'category': None
            }), 200
        
        color_info = get_aqi_color(aqi)
        
        return jsonify({
            'inside': True,
            'aqi': aqi,
            'category': color_info['category'],
            'label': color_info['label'],
            'color': color_info['color']
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting AQI value: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)