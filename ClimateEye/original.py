from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta, timezone
import openmeteo_requests
import requests_cache
from retry_requests import retry
import logging,os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_next_hour_record(hourly_records, current_hour):
    for i, record in enumerate(hourly_records):
        if record["date"] == current_hour["date"]:
            if i + 1 < len(hourly_records):
                return hourly_records[i + 1]
    return None

def get_current_hour_record(hourly_records):
    """
    Select the most recent hour <= current LOCAL time
    (AQI data timestamps are local-time based)
    """
    now = datetime.now()  # LOCAL server time

    valid_records = []

    for record in hourly_records:
        try:
            record_time = datetime.strptime(
                record["date"],
                "%Y-%m-%d %H:%M:%S"
            )

            if record_time <= now:
                valid_records.append((record_time, record))

        except Exception:
            continue

    if not valid_records:
        return None

    valid_records.sort(key=lambda x: x[0])
    return valid_records[-1][1]

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
        
        # Check if horizontal ray from point crosses edge
        # Ray goes from (lat, lon) to (lat, infinity)
        # Edge goes from (xi, yi) to (xj, yj)
        
        # Check if ray crosses this edge
        if ((yi > lon) != (yj > lon)):  # Edge crosses horizontal line at point's longitude
            # Calculate x-coordinate of intersection
            if yj != yi:  # Avoid division by zero
                x_intersect = (lon - yi) * (xj - xi) / (yj - yi) + xi
                if lat < x_intersect:  # Point is to the left of intersection
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
    import math
    
    # Validate bounds (purely geometric validation)
    if max_lat <= min_lat or max_lon <= min_lon:
        return []
    
    # Coordinate-to-distance conversion factors (purely geometric, no directional bias)
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


def get_relative_humidity(latitude, longitude, target_datetime):
    """
    Fetch relative humidity from weather API for a specific datetime
    Uses the same logic as weather/hourly endpoint to ensure exact matching
    Returns RH as percentage (0-100) or None if not available
    """
    try:
        # Parse target datetime to get date
        if isinstance(target_datetime, str):
            dt = datetime.strptime(target_datetime, '%Y-%m-%d %H:%M:%S')
            # Convert to UTC if not timezone-aware
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = target_datetime
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        
        date_str = dt.strftime('%Y-%m-%d')
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ["relative_humidity_2m"],
            "start_date": date_str,
            "end_date": date_str,
            "timezone": "auto"
        }
        
        # For today, request tomorrow too to get full forecast (same as weather/hourly)
        today = datetime.now().date()
        is_today = False
        try:
            request_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            is_today = request_date == today
            if is_today:
                tomorrow = (request_date + timedelta(days=1)).strftime('%Y-%m-%d')
                params["end_date"] = tomorrow
        except ValueError:
            pass
        
        responses = openmeteo.weather_api(url, params=params)
        
        if not responses:
            return None
        
        response = responses[0]
        hourly = response.Hourly()
        hourly_humidity = hourly.Variables(0).ValuesAsNumpy()
        
        # Get hourly times - use same logic as weather/hourly endpoint
        try:
            start_time = float(hourly.Time())
            num_values = len(hourly_humidity)
            try:
                interval = float(hourly.Interval())
            except (AttributeError, TypeError, ValueError):
                interval = 3600  # Default to 1 hour
            hourly_times = [start_time + (i * interval) for i in range(num_values)]
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error getting hourly times: {str(e)}")
            try:
                hourly_times_array = hourly.Time()
                if hasattr(hourly_times_array, '__iter__') and not isinstance(hourly_times_array, (str, bytes)):
                    hourly_times = list(hourly_times_array)
                else:
                    hourly_times = []
            except Exception as e2:
                logger.error(f"Error in fallback time parsing: {str(e2)}")
                hourly_times = []
        
        # Match exact hour - use same logic as weather/hourly endpoint
        # Normalize target datetime to hour boundary (minute=0, second=0, microsecond=0)
        target_hour = dt.replace(minute=0, second=0, microsecond=0)
        target_hour_timestamp = target_hour.timestamp()
        
        # Find exact match by comparing timestamps (within 1 second tolerance for rounding)
        # This ensures we get the exact same humidity that weather/hourly would return
        for i in range(len(hourly_times)):
            try:
                hour_timestamp = float(hourly_times[i])
                hour_datetime = datetime.fromtimestamp(hour_timestamp, tz=timezone.utc)
                
                # Match exact hour by comparing timestamps (within 1 second tolerance)
                # Also verify date/hour components match as double-check
                timestamp_diff = abs(hour_timestamp - target_hour_timestamp)
                if (timestamp_diff <= 1.0 and  # Within 1 second
                    hour_datetime.year == target_hour.year and
                    hour_datetime.month == target_hour.month and
                    hour_datetime.day == target_hour.day and
                    hour_datetime.hour == target_hour.hour):
                    if i < len(hourly_humidity):
                        rh_value = float(hourly_humidity[i])
                        if rh_value != rh_value:  # NaN check
                            return None
                        return rh_value
            except (TypeError, ValueError, OSError):
                continue
        
        # Fallback: find closest match within 1 hour (should rarely be needed)
        closest_index = -1
        min_diff = float('inf')
        for i, hour_timestamp in enumerate(hourly_times):
            try:
                hour_ts = float(hour_timestamp)
                diff = abs(hour_ts - target_hour_timestamp)
                # Only consider matches within 1 hour
                if diff < 3600 and diff < min_diff:
                    min_diff = diff
                    closest_index = i
            except (TypeError, ValueError):
                continue
        
        if closest_index >= 0 and closest_index < len(hourly_humidity):
            rh_value = float(hourly_humidity[closest_index])
            if rh_value != rh_value:  # NaN check
                return None
            return rh_value
        
        return None
    except Exception as e:
        logger.error(f"Error fetching relative humidity: {str(e)}")
        return None

def calculate_correction_factor(latitude, longitude, target_datetime, rh=None):
    """
    Calculate Correction Factor (CF) for AQI correction
    CF = VPC × TDC × LSAF × HCF
    
    Parameters:
    - VPC = 1 + α, where α = 1.5 (default) → VPC = 2.5
    - TDC = 1.1 for 11am-6pm, else 1.45
    - LSAF = 1 + β, where β = 0.3 (default) → LSAF = 1.3
    - HCF = 1 + γ × (RH - 40) / 60, where γ = 0.3
    
    Returns CF value
    """
    # VPC (Vertical Proximity Correction)
    alpha = 1.5
    vpc = 1 + alpha  # VPC = 2.5
    
    # TDC (Time-of-day Correction)
    if isinstance(target_datetime, str):
        dt = datetime.strptime(target_datetime, '%Y-%m-%d %H:%M:%S')
    else:
        dt = target_datetime
    
    hour = dt.hour
    # 11am to 6pm (11:00 to 18:59)
    if 11 <= hour < 18:
        tdc = 1.1
    else:
        tdc = 1.45
    
    # LSAF (Location Specific Adjustment Factor)
    beta = 0.3
    lsaf = 1 + beta  # LSAF = 1.3
    
    # HCF (Humidity Correction Factor)
    if rh is None:
        # Fetch RH if not provided
        rh = get_relative_humidity(latitude, longitude, dt)
    
    if rh is not None:
        gamma = 0.3
        hcf = 1 + gamma * (rh - 40) / 60
    else:
        # Default HCF if RH not available (assume RH = 50%)
        gamma = 0.3
        hcf = 1 + gamma * (50 - 40) / 60  # HCF = 1.05
    
    # Calculate CF
    cf = vpc * tdc * lsaf * hcf
    
    return {
        "cf": round(cf, 4),
        "vpc": vpc,
        "tdc": tdc,
        "lsaf": lsaf,
        "hcf": round(hcf, 4) if rh is not None else round(1.05, 4),
        "rh": round(rh, 2) if rh is not None else None
    }

def compress_hourly_records(hourly_records, sample_every=2, use_0to3m=False):
    """
    Compress hourly records to reduce token usage:
    - Only include essential fields
    - Sample every N hours (default: every 2 hours)
    - Format compactly
    - If use_0to3m=True, use 0to3m parameters instead of raw
    """
    if use_0to3m:
        essential_fields = ['date', 'aqi_0to3m', 'pm2_5_0to3m', 'pm10_0to3m', 'o3_0to3m', 'co_0to3m', 'no2_0to3m', 'so2_0to3m']
    else:
        essential_fields = ['date', 'aqi', 'pm2_5', 'pm10', 'o3', 'co', 'no2', 'so2']
    compressed = []
    
    for i, record in enumerate(hourly_records):
        # Sample every N hours, but always include first and last
        if i == 0 or i == len(hourly_records) - 1 or i % sample_every == 0:
            compressed_record = {k: record.get(k) for k in essential_fields if k in record}
            compressed.append(compressed_record)
    
    return compressed

def format_hourly_trend_compact(hourly_records, use_0to3m=False):
    """Format hourly records in a compact string format"""
    if not hourly_records:
        return "No data"
    
    lines = []
    for record in hourly_records:
        time_str = record.get('date', 'N/A')
        if use_0to3m:
            aqi = record.get('aqi_0to3m', 'N/A')
            pm25 = record.get('pm2_5_0to3m', 'N/A')
            pm10 = record.get('pm10_0to3m', 'N/A')
        else:
            aqi = record.get('aqi', 'N/A')
            pm25 = record.get('pm2_5', 'N/A')
            pm10 = record.get('pm10', 'N/A')
        lines.append(f"{time_str}: AQI={aqi}, PM2.5={pm25}, PM10={pm10}")
    
    return "\n".join(lines)

def build_strict_time_prompt(current_hour, hourly_records, next_hour=None, use_0to3m=False):
    # Compress hourly records to reduce token usage
    compressed_records = compress_hourly_records(hourly_records, sample_every=2, use_0to3m=use_0to3m)
    trend_text = format_hourly_trend_compact(compressed_records, use_0to3m=use_0to3m)
    
    # Compress next hour if it exists
    next_hour_text = "None"
    if next_hour:
        if use_0to3m:
            next_hour_text = f"Time: {next_hour.get('date', 'N/A')}, AQI: {next_hour.get('aqi_0to3m', 'N/A')}, PM2.5: {next_hour.get('pm2_5_0to3m', 'N/A')}, PM10: {next_hour.get('pm10_0to3m', 'N/A')}"
        else:
            next_hour_text = f"Time: {next_hour.get('date', 'N/A')}, AQI: {next_hour.get('aqi', 'N/A')}, PM2.5: {next_hour.get('pm2_5', 'N/A')}, PM10: {next_hour.get('pm10', 'N/A')}"
    
    return f"""
You are professional Air quality officer with 30 years of experience in climate and environment monitoring.
You also have expertise in satellite based AQI quantification, predicting root causes for AQI changes

STRICT RULES (MANDATORY):
- Treat CURRENT HOUR DATA as the SINGLE SOURCE OF TRUTH.
- Do NOT refer to any other hour as current.
- Use FULL DAY TREND only to explain patterns and causes.
- CRITICAL: Any hour in the FULL DAY TREND that is LATER than the CURRENT HOUR time must be treated as a PREDICTION, not observed data.
- When referencing future hours, ALWAYS use prediction language: "is expected to", "may reach", "is predicted to", "could peak at".
- NEVER describe future values as already observed or as facts.
- Do NOT invent values, times, or causes.

===========================
CURRENT HOUR DATA (SOURCE OF TRUTH):
Time: {current_hour['date']}
AQI: {current_hour['aqi_0to3m' if use_0to3m else 'aqi']}
PM2.5: {current_hour['pm2_5_0to3m' if use_0to3m else 'pm2_5']} µg/m³
PM10: {current_hour['pm10_0to3m' if use_0to3m else 'pm10']} µg/m³
O₃: {current_hour['o3_0to3m' if use_0to3m else 'o3']} µg/m³
CO: {current_hour['co_0to3m' if use_0to3m else 'co']} µg/m³
NO₂: {current_hour['no2_0to3m' if use_0to3m else 'no2']} µg/m³
SO₂: {current_hour['so2_0to3m' if use_0to3m else 'so2']} µg/m³
===========================

FULL DAY TREND (FOR CONTEXT ONLY — PAST + PREDICTED):
{trend_text}
NOTE: Hours LATER than the CURRENT HOUR are PREDICTIONS, not observed data. Always use prediction language when referencing them.

NEXT HOUR PREDICTION (OPTIONAL — FUTURE ONLY):
{next_hour_text}

Explain using EXACTLY the following headings
(do NOT add or remove headings):

1. Why AQI is elevated
   - Refer to the CURRENT HOUR first.
   - Use the daily trend ONLY to say whether pollution is:
     • persistent throughout the day (based on PAST hours only)
     • or time-specific (morning / afternoon / evening) based on PAST patterns.
   - If referencing future hours in the trend, clearly label them as PREDICTIONS (e.g., "pollution is predicted to peak in the afternoon").
   - Do NOT present future hours as observed facts.

2. Main contributing pollutants
   - Identify the PRIME pollutant for the CURRENT HOUR.
   - Support this using:
     • current concentrations
     • daily averages or past peaks from hours BEFORE the current hour.
   - If referencing future hours, ALWAYS label as predictions (e.g., "PM2.5 is predicted to reach 65.2 µg/m³ at 20:00").
   - Do NOT use future peaks as if they are already observed.

3. Why those pollutants increase
   - Explain realistic sources:
     • traffic (petrol vs diesel)
     • industry
     • construction / dust
     • burning
     • weather conditions
   - Tie causes to time-of-day patterns seen in the PAST data only (hours BEFORE current hour).
   - If referencing future hours, use prediction language (e.g., "construction activity is expected to increase in the afternoon").

4. How they raise AQI
   - Explain how the pollutant concentration range maps to AQI categories.
   - Clearly state which AQI threshold is being crossed:
     Good → Moderate → Poor → Unhealthy.

5. Mitigation (individual + city level) — WITH NUMBERS
   - Use ROUGH, REALISTIC estimates (ranges, %, counts).
   - Examples (estimate based on the data, do NOT copy blindly):
     • Reducing ~10,000 vehicles/day may lower PM2.5 by ~5–10 µg/m³
     • A 30–40% cut in diesel traffic can shift AQI down by one category
     • Eliminating open burning can reduce PM2.5 by ~15–25%
   - Give government approved mitigation strategy based on identified cause particularly for construction works
   - Add specific time zone for action so that mitigation strategy can be highly effective
   - If referencing future hours for time-specific actions, use prediction language (e.g., "restrict activities during predicted peak hours 14:00-18:00").
   - Separate clearly:
     a) Individual actions
     b) City / policy actions

6. Short outlook for next hour (ONLY if prediction exists)
   - Clearly label this section as prediction.
   - Use phrases like:
     "is expected to", "may increase", "is predicted".
   - Do NOT present predictions as facts.

Rules:
- Use bullet points.
- Use numeric ranges, percentages, and approximate counts.
- Be factual, practical, and time-consistent.
- Do NOT exaggerate certainty.
"""


app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup the Open-Meteo API client with cache and retry on error
# Reduced cache time to 5 minutes for more real-time data
cache_session = requests_cache.CachedSession('.cache', expire_after=300)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


def calculate_aqi(pm25, pm10, o3=None, no2=None, so2=None, co=None):
    """
    Calculate AQI-US based on all available pollutants
    Uses the US EPA AQI calculation method
    The overall AQI is the maximum of all pollutant sub-indices
    """
    def calculate_aqi_for_pollutant(concentration, breakpoints):
        """Calculate AQI for a single pollutant"""
        if concentration is None:
            return None
        try:
            concentration = float(concentration)
        except (ValueError, TypeError):
            return None
        
        # Handle values below the first breakpoint
        if concentration < breakpoints[0][0]:
            return 0
        # Handle values above the last breakpoint
        if concentration > breakpoints[-1][1]:
            return 500
        # Find the appropriate breakpoint range
        for i in range(len(breakpoints)):
            # Check if concentration is within this range (inclusive)
            if breakpoints[i][0] <= concentration <= breakpoints[i][1]:
                aqi_low = breakpoints[i][2]
                aqi_high = breakpoints[i][3]
                conc_low = breakpoints[i][0]
                conc_high = breakpoints[i][1]
                if conc_high == conc_low:
                    return aqi_low
                aqi = ((aqi_high - aqi_low) / (conc_high - conc_low)) * (concentration - conc_low) + aqi_low
                return round(aqi)
        # If no range found (shouldn't happen), return None
        return None
    
    # PM2.5 breakpoints (24-hour average, µg/m³): [conc_low, conc_high, aqi_low, aqi_high]
    pm25_breakpoints = [
        [0, 12.0, 0, 50],      # Good
        [12.1, 35.4, 51, 100],  # Moderate
        [35.5, 55.4, 101, 150], # Unhealthy for Sensitive
        [55.5, 150.4, 151, 200], # Unhealthy
        [150.5, 250.4, 201, 300], # Very Unhealthy
        [250.5, 500.4, 301, 500]  # Hazardous
    ]
    
    # PM10 breakpoints (24-hour average, µg/m³)
    pm10_breakpoints = [
        [0, 54, 0, 50],
        [55, 154, 51, 100],
        [155, 254, 101, 150],
        [255, 354, 151, 200],
        [355, 424, 201, 300],
        [425, 604, 301, 500]
    ]
    
    # O3 breakpoints (8-hour average, µg/m³)
    # Open-Meteo provides O3 in µg/m³, EPA uses ppm, so we use µg/m³ directly
    # Conversion: 1 ppm = 1960 µg/m³ at 25°C
    # EPA breakpoints in ppm: 0-0.054, 0.055-0.070, 0.071-0.085, 0.086-0.105, 0.106-0.200
    # In µg/m³: 0-105.84, 107.8-137.2, 139.16-166.6, 168.56-205.8, 207.76-392
    o3_breakpoints = [
        [0, 105.84, 0, 50],
        [107.8, 137.2, 51, 100],
        [139.16, 166.6, 101, 150],
        [168.56, 205.8, 151, 200],
        [207.76, 392, 201, 300],
        [392.1, 980, 301, 500]  # Extended for higher values
    ]
    
    # NO2 breakpoints (1-hour average, µg/m³)
    # EPA uses ppb, conversion: 1 ppb = 1.88 µg/m³ at 25°C
    # EPA breakpoints in ppb: 0-53, 54-100, 101-360, 361-649, 650-1249, 1250-2049
    # In µg/m³: 0-99.64, 101.52-188, 189.88-676.8, 678.68-1220.12, 1222-2348.12, 2349-3852.12
    no2_breakpoints = [
        [0, 99.64, 0, 50],
        [101.52, 188, 51, 100],
        [189.88, 676.8, 101, 150],
        [678.68, 1220.12, 151, 200],
        [1222, 2348.12, 201, 300],
        [2349, 3852.12, 301, 500]
    ]
    
    # SO2 breakpoints (1-hour average, µg/m³)
    # EPA uses ppb, conversion: 1 ppb = 2.62 µg/m³ at 25°C
    # EPA breakpoints in ppb: 0-35, 36-75, 76-185, 186-304, 305-604, 605-1004
    # In µg/m³: 0-91.7, 94.32-196.5, 199.12-484.7, 487.32-796.48, 799.1-1582.48, 1585.1-2630.48
    so2_breakpoints = [
        [0, 91.7, 0, 50],
        [94.32, 196.5, 51, 100],
        [199.12, 484.7, 101, 150],
        [487.32, 796.48, 151, 200],
        [799.1, 1582.48, 201, 300],
        [1585.1, 2630.48, 301, 500]
    ]
    
    # CO breakpoints (8-hour average, µg/m³)
    # EPA uses ppm, conversion: 1 ppm = 1145 µg/m³ at 25°C
    # EPA breakpoints in ppm: 0-4.4, 4.5-9.4, 9.5-12.4, 12.5-15.4, 15.5-30.4, 30.5-50.4
    # In µg/m³: 0-5038, 5152.5-10763, 10877.5-14198, 14312.5-17633, 17747.5-34808, 34922.5-57708
    co_breakpoints = [
        [0, 5038, 0, 50],
        [5152.5, 10763, 51, 100],
        [10877.5, 14198, 101, 150],
        [14312.5, 17633, 151, 200],
        [17747.5, 34808, 201, 300],
        [34922.5, 57708, 301, 500]
    ]
    
    # Calculate AQI for each pollutant
    aqi_values = []
    
    if pm25 is not None:
        aqi_pm25 = calculate_aqi_for_pollutant(pm25, pm25_breakpoints)
        if aqi_pm25 is not None:
            aqi_values.append(aqi_pm25)
    
    if pm10 is not None:
        aqi_pm10 = calculate_aqi_for_pollutant(pm10, pm10_breakpoints)
        if aqi_pm10 is not None:
            aqi_values.append(aqi_pm10)
    
    if o3 is not None:
        aqi_o3 = calculate_aqi_for_pollutant(o3, o3_breakpoints)
        if aqi_o3 is not None:
            aqi_values.append(aqi_o3)
    
    if no2 is not None:
        aqi_no2 = calculate_aqi_for_pollutant(no2, no2_breakpoints)
        if aqi_no2 is not None:
            aqi_values.append(aqi_no2)
    
    if so2 is not None:
        aqi_so2 = calculate_aqi_for_pollutant(so2, so2_breakpoints)
        if aqi_so2 is not None:
            aqi_values.append(aqi_so2)
    
    if co is not None:
        aqi_co = calculate_aqi_for_pollutant(co, co_breakpoints)
        if aqi_co is not None:
            aqi_values.append(aqi_co)
    
    # Return the maximum AQI value (overall AQI is the highest sub-index)
    if aqi_values:
        return max(aqi_values)
    else:
        return None


@app.route('/api/aqi', methods=['POST'])
def get_aqi():
    """
    Get Air Quality Index data from Open-Meteo
    Expects: { "latitude": float, "longitude": float, "date": "YYYY-MM-DD" }
    """
    try:
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        date = data.get('date')  # Optional: specific date
        
        if not latitude or not longitude:
            return jsonify({'error': 'Latitude and longitude are required'}), 400
        
        # Parse date if provided
        start_date = None
        end_date = None
        if date:
            try:
                date_obj = datetime.strptime(date, '%Y-%m-%d')
                start_date = date_obj.strftime('%Y-%m-%d')
                end_date = date_obj.strftime('%Y-%m-%d')
            except ValueError:
                return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", 
                      "sulphur_dioxide", "ozone"],
        }
        
        if start_date and end_date:
            params["start_date"] = start_date
            params["end_date"] = end_date
        
        responses = openmeteo.weather_api(url, params=params)
        
        if not responses:
            return jsonify({'error': 'No data available for the specified location'}), 404
        
        response = responses[0]
        
        # Check if this is a historical date (before today)
        today = datetime.now().date()
        is_historical = False
        if date:
            try:
                request_date = datetime.strptime(date, '%Y-%m-%d').date()
                is_historical = request_date < today
            except ValueError:
                is_historical = False
        
        if is_historical:
            # For historical dates, use hourly data and calculate daily average
            hourly = response.Hourly()
            hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
            hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()
            hourly_carbon_monoxide = hourly.Variables(2).ValuesAsNumpy()
            hourly_nitrogen_dioxide = hourly.Variables(3).ValuesAsNumpy()
            hourly_sulphur_dioxide = hourly.Variables(4).ValuesAsNumpy()
            hourly_ozone = hourly.Variables(5).ValuesAsNumpy()
            
            # Calculate daily averages (handle empty arrays and NaN values)
            def safe_mean(arr):
                if len(arr) == 0:
                    return None
                try:
                    mean_val = float(arr.mean())
                    # Check for NaN
                    if mean_val != mean_val:  # NaN check
                        return None
                    return mean_val
                except (ValueError, TypeError, AttributeError):
                    return None
            
            avg_pm10 = safe_mean(hourly_pm10)
            avg_pm2_5 = safe_mean(hourly_pm2_5)
            avg_carbon_monoxide = safe_mean(hourly_carbon_monoxide)
            avg_nitrogen_dioxide = safe_mean(hourly_nitrogen_dioxide)
            avg_sulphur_dioxide = safe_mean(hourly_sulphur_dioxide)
            avg_ozone = safe_mean(hourly_ozone)
            
            # Calculate AQI from averages (including all pollutants)
            aqi = calculate_aqi(
                pm25=avg_pm2_5,
                pm10=avg_pm10,
                o3=avg_ozone,
                no2=avg_nitrogen_dioxide,
                so2=avg_sulphur_dioxide,
                co=avg_carbon_monoxide
            )
            

            # Calculate correction factor for daily average (use noon time)
            # Create datetime object for noon of the date in UTC
            date_time_str = date + " 12:00:00"
            date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
            cf_data = calculate_correction_factor(latitude, longitude, date_time_obj)
            cf = cf_data["cf"]
            
            # Calculate corrected values
            pm10_0to3m = round(avg_pm10 * cf, 2) if avg_pm10 is not None else None
            pm2_5_0to3m = round(avg_pm2_5 * cf, 2) if avg_pm2_5 is not None else None
            co_0to3m = round(avg_carbon_monoxide * cf, 2) if avg_carbon_monoxide is not None else None
            no2_0to3m = round(avg_nitrogen_dioxide * cf, 2) if avg_nitrogen_dioxide is not None else None
            so2_0to3m = round(avg_sulphur_dioxide * cf, 2) if avg_sulphur_dioxide is not None else None
            o3_0to3m = round(avg_ozone * cf, 2) if avg_ozone is not None else None
            
            aqi_corrected = calculate_aqi(
                    pm25=pm2_5_0to3m,
                    pm10=pm10_0to3m,
                    o3=o3_0to3m,
                    no2=no2_0to3m,
                    so2=so2_0to3m,
                    co=co_0to3m
                )

            # Calculate corrected AQI by multiplying raw AQI with correction factor
            aqi_0to3m = round(aqi * cf, 2) if aqi is not None else None
            
            result = {
                "date": date_time_str,
                "aqi": int(aqi) if aqi is not None else None,
                "aqi_corrected": int(aqi_corrected) if aqi_corrected is not None else None,
                "pm10": round(avg_pm10, 2) if avg_pm10 is not None else None,
                "pm2_5": round(avg_pm2_5, 2) if avg_pm2_5 is not None else None,
                "co": round(avg_carbon_monoxide, 2) if avg_carbon_monoxide is not None else None,
                "no2": round(avg_nitrogen_dioxide, 2) if avg_nitrogen_dioxide is not None else None,
                "so2": round(avg_sulphur_dioxide, 2) if avg_sulphur_dioxide is not None else None,
                "o3": round(avg_ozone, 2) if avg_ozone is not None else None,
                # Corrected values
                "aqi_0to3m": int(round(aqi_0to3m)) if aqi_0to3m is not None else None,
                "pm10_0to3m": pm10_0to3m,
                "pm2_5_0to3m": pm2_5_0to3m,
                "co_0to3m": co_0to3m,
                "no2_0to3m": no2_0to3m,
                "so2_0to3m": so2_0to3m,
                "o3_0to3m": o3_0to3m,
                # Correction factor details
                "correction_factor": {
                    "cf": cf,
                    "vpc": cf_data["vpc"],
                    "tdc": cf_data["tdc"],
                    "lsaf": cf_data["lsaf"],
                    "hcf": cf_data["hcf"],
                    "rh": cf_data["rh"]
                }
            }
        else:
            # For current date, use hourly data and get the most recent complete hour
            # Initialize hour_datetime to current time as fallback
            now_init = datetime.now(timezone.utc)
            current_hour_init = now_init.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            hour_datetime = current_hour_init
            
            hourly = response.Hourly()
            hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
            hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()
            hourly_carbon_monoxide = hourly.Variables(2).ValuesAsNumpy()
            hourly_nitrogen_dioxide = hourly.Variables(3).ValuesAsNumpy()
            hourly_sulphur_dioxide = hourly.Variables(4).ValuesAsNumpy()
            hourly_ozone = hourly.Variables(5).ValuesAsNumpy()
            
            # Get hourly times - Open-Meteo returns Unix timestamps in seconds
            # hourly.Time() might return a single start timestamp or an array
            # We'll calculate timestamps from start time and interval if needed
            try:
                start_time = float(hourly.Time())
                num_values = len(hourly_pm10)
                
                # Try to get interval, default to 3600 seconds (1 hour) if not available
                try:
                    interval = float(hourly.Interval())  # Interval in seconds
                except (AttributeError, TypeError, ValueError):
                    interval = 3600  # Default to 1 hour
                
                # Calculate all timestamps from start time and interval
                hourly_times = [start_time + (i * interval) for i in range(num_values)]
            except (AttributeError, TypeError, ValueError) as e:
                logger.error(f"Error getting hourly times: {str(e)}")
                # Fallback: try to get as array
                try:
                    hourly_times_array = hourly.Time()
                    if hasattr(hourly_times_array, '__iter__') and not isinstance(hourly_times_array, (str, bytes)):
                        hourly_times = list(hourly_times_array)
                    else:
                        hourly_times = []
                except Exception as e2:
                    logger.error(f"Error in fallback time parsing: {str(e2)}")
                    hourly_times = []
            
            # Get current time and find the most recent complete hour
            now = datetime.now(timezone.utc)
            # Get the most recent complete hour (subtract 1 hour from current hour)
            current_hour = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            current_hour_timestamp = current_hour.timestamp()
            
            # Find the index of the most recent complete hour (not current or future)
            latest_index = -1
            if len(hourly_times) > 0:
                # Iterate backwards to find the most recent hour <= current_hour
                for i in range(len(hourly_times) - 1, -1, -1):
                    try:
                        # hourly.Time() returns Unix timestamps in seconds
                        hour_timestamp = float(hourly_times[i])
                        hour_datetime = datetime.fromtimestamp(hour_timestamp, tz=timezone.utc)
                        
                        # Use the most recent hour that is <= current_hour (most recent complete hour)
                        if hour_timestamp <= current_hour_timestamp:
                            latest_index = i
                            break
                    except (IndexError, TypeError, ValueError, OSError) as e:
                        logger.error(f"Error parsing hour time at index {i}: {str(e)}")
                        continue
            
            # If no valid hour found, use the last available (simpler fallback)
            if latest_index == -1:
                if len(hourly_pm10) > 0:
                    latest_index = len(hourly_pm10) - 1
                else:
                    latest_index = -1
            
            # Get values for the most recent complete hour
            if latest_index >= 0 and latest_index < len(hourly_pm10):
                try:
                    def safe_float(value):
                        if value is None:
                            return None
                        try:
                            val = float(value)
                            # Check for NaN
                            if val != val:  # NaN check
                                return None
                            return val
                        except (TypeError, ValueError):
                            return None
                    
                    pm10 = safe_float(hourly_pm10[latest_index])
                    pm2_5 = safe_float(hourly_pm2_5[latest_index])
                    carbon_monoxide = safe_float(hourly_carbon_monoxide[latest_index])
                    nitrogen_dioxide = safe_float(hourly_nitrogen_dioxide[latest_index])
                    sulphur_dioxide = safe_float(hourly_sulphur_dioxide[latest_index])
                    ozone = safe_float(hourly_ozone[latest_index])
                except (IndexError, TypeError, ValueError) as e:
                    logger.error(f"Error accessing hourly AQI data at index {latest_index}: {str(e)}")
                    pm10 = None
                    pm2_5 = None
                    carbon_monoxide = None
                    nitrogen_dioxide = None
                    sulphur_dioxide = None
                    ozone = None
                
                # Get the timestamp for this hour
                try:
                    if latest_index < len(hourly_times):
                        # hourly.Time() returns Unix timestamps in seconds
                        hour_timestamp = float(hourly_times[latest_index])
                        hour_datetime = datetime.fromtimestamp(hour_timestamp, tz=timezone.utc)
                    else:
                        hour_datetime = current_hour
                except (IndexError, TypeError, ValueError, OSError):
                    hour_datetime = current_hour
            else:
                # Fallback if no data available
                pm10 = None
                pm2_5 = None
                carbon_monoxide = None
                nitrogen_dioxide = None
                sulphur_dioxide = None
                ozone = None
                hour_datetime = current_hour
            
            # Calculate AQI (including all pollutants)
            aqi = calculate_aqi(
                pm25=pm2_5,
                pm10=pm10,
                o3=ozone,
                no2=nitrogen_dioxide,
                so2=sulphur_dioxide,
                co=carbon_monoxide
            )
            
            # Calculate correction factor for this specific hour
            # Pass the datetime object directly to ensure exact matching with weather/hourly
            hour_datetime_str = hour_datetime.strftime('%Y-%m-%d %H:%M:%S')
            cf_data = calculate_correction_factor(latitude, longitude, hour_datetime)
            cf = cf_data["cf"]
            
            # Calculate corrected values
            pm10_0to3m = round(pm10 * cf, 2) if pm10 is not None else None
            pm2_5_0to3m = round(pm2_5 * cf, 2) if pm2_5 is not None else None
            co_0to3m = round(carbon_monoxide * cf, 2) if carbon_monoxide is not None else None
            no2_0to3m = round(nitrogen_dioxide * cf, 2) if nitrogen_dioxide is not None else None
            so2_0to3m = round(sulphur_dioxide * cf, 2) if sulphur_dioxide is not None else None
            o3_0to3m = round(ozone * cf, 2) if ozone is not None else None
            
            # Calculate corrected AQI by multiplying raw AQI with correction factor
            aqi_0to3m = round(aqi * cf, 2) if aqi is not None else None
            
            # For "Last Updated", use current time, not data timestamp
            current_time = datetime.now(timezone.utc)
            
            result = {
                "date": hour_datetime_str,  # Data timestamp
                "last_updated": current_time.strftime('%Y-%m-%d %H:%M:%S'),  # Current time
                "aqi": int(aqi) if aqi is not None else None,
                "pm10": round(pm10, 2) if pm10 is not None else None,
                "pm2_5": round(pm2_5, 2) if pm2_5 is not None else None,
                "co": round(carbon_monoxide, 2) if carbon_monoxide is not None else None,
                "no2": round(nitrogen_dioxide, 2) if nitrogen_dioxide is not None else None,
                "so2": round(sulphur_dioxide, 2) if sulphur_dioxide is not None else None,
                "o3": round(ozone, 2) if ozone is not None else None,
                # Corrected values
                "aqi_0to3m": int(round(aqi_0to3m)) if aqi_0to3m is not None else None,
                "pm10_0to3m": pm10_0to3m,
                "pm2_5_0to3m": pm2_5_0to3m,
                "co_0to3m": co_0to3m,
                "no2_0to3m": no2_0to3m,
                "so2_0to3m": so2_0to3m,
                "o3_0to3m": o3_0to3m,
                # Correction factor details
                "correction_factor": {
                    "cf": cf,
                    "vpc": cf_data["vpc"],
                    "tdc": cf_data["tdc"],
                    "lsaf": cf_data["lsaf"],
                    "hcf": cf_data["hcf"],
                    "rh": cf_data["rh"]
                }
            }
        
        return jsonify(result), 200
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error fetching AQI data: {str(e)}")
        logger.error(f"Traceback: {error_trace}")
        # Return a more user-friendly error message
        return jsonify({
            'error': f'Failed to fetch AQI data: {str(e)}',
            'details': 'Please check the backend logs for more information'
        }), 500


@app.route('/api/aqi/hourly', methods=['POST'])
def get_aqi_hourly():
    """
    Get hourly Air Quality Index data from Open-Meteo for a specific date
    Expects: { "latitude": float, "longitude": float, "date": "YYYY-MM-DD" }
    Returns: Array of hourly AQI records with trend indicators
    """
    try:
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        date = data.get('date')  # Required: specific date
        
        if not latitude or not longitude:
            return jsonify({'error': 'Latitude and longitude are required'}), 400
        
        if not date:
            return jsonify({'error': 'Date is required'}), 400
        
        try:
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            start_date = date_obj.strftime('%Y-%m-%d')
            end_date = date_obj.strftime('%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", 
                      "sulphur_dioxide", "ozone"],
            "start_date": start_date,
            "end_date": end_date,
        }
        
        # For today, ensure we get forecast data by requesting up to tomorrow
        today = datetime.now().date()
        try:
            request_date = datetime.strptime(date, '%Y-%m-%d').date()
            if request_date == today:
                # Request data for today and tomorrow to get full forecast
                tomorrow = (request_date + timedelta(days=1)).strftime('%Y-%m-%d')
                params["end_date"] = tomorrow
        except ValueError:
            pass
        
        responses = openmeteo.weather_api(url, params=params)
        
        if not responses:
            return jsonify({'error': 'No data available for the specified location'}), 404
        
        response = responses[0]
        hourly = response.Hourly()
        hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
        hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()
        hourly_carbon_monoxide = hourly.Variables(2).ValuesAsNumpy()
        hourly_nitrogen_dioxide = hourly.Variables(3).ValuesAsNumpy()
        hourly_sulphur_dioxide = hourly.Variables(4).ValuesAsNumpy()
        hourly_ozone = hourly.Variables(5).ValuesAsNumpy()
        
        # Get hourly times
        try:
            start_time = float(hourly.Time())
            num_values = len(hourly_pm10)
            
            try:
                interval = float(hourly.Interval())
            except (AttributeError, TypeError, ValueError):
                interval = 3600  # Default to 1 hour
            
            hourly_times = [start_time + (i * interval) for i in range(num_values)]
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error getting hourly times: {str(e)}")
            try:
                hourly_times_array = hourly.Time()
                if hasattr(hourly_times_array, '__iter__') and not isinstance(hourly_times_array, (str, bytes)):
                    hourly_times = list(hourly_times_array)
                else:
                    hourly_times = []
            except Exception as e2:
                logger.error(f"Error in fallback time parsing: {str(e2)}")
                hourly_times = []
        
        # Check if this is today
        today = datetime.now().date()
        is_today = False
        try:
            request_date = datetime.strptime(date, '%Y-%m-%d').date()
            is_today = request_date == today
        except ValueError:
            pass
        
        # Get current time for today (in UTC)
        now = datetime.now(timezone.utc)
        current_time_timestamp = now.timestamp()
        
        def safe_float(value):
            if value is None:
                return None
            try:
                val = float(value)
                if val != val:  # NaN check
                    return None
                return val
            except (TypeError, ValueError):
                return None
        
        # Build hourly records
        hourly_records = []
        previous_aqi = None
        
        # Filter to only show hours for the requested date
        requested_date_obj = datetime.strptime(date, '%Y-%m-%d').date()
        # Create date boundaries for filtering (start and end of requested date in UTC)
        requested_date_start = datetime.combine(requested_date_obj, datetime.min.time()).replace(tzinfo=timezone.utc)
        requested_date_end = requested_date_start + timedelta(days=1)
        
        for i in range(len(hourly_pm10)):
            try:
                hour_timestamp = float(hourly_times[i]) if i < len(hourly_times) else None
                if hour_timestamp:
                    hour_datetime = datetime.fromtimestamp(hour_timestamp, tz=timezone.utc)
                else:
                    continue
                
                # Only include hours that belong to the requested date
                # Check if hour is within the requested date range (00:00 to 23:59 of that date)
                if hour_datetime < requested_date_start or hour_datetime >= requested_date_end:
                    continue
                
                # Show all hours for the requested date (including forecast for today)
                
                pm10 = safe_float(hourly_pm10[i])
                pm2_5 = safe_float(hourly_pm2_5[i])
                carbon_monoxide = safe_float(hourly_carbon_monoxide[i])
                nitrogen_dioxide = safe_float(hourly_nitrogen_dioxide[i])
                sulphur_dioxide = safe_float(hourly_sulphur_dioxide[i])
                ozone = safe_float(hourly_ozone[i])
                
                # Calculate AQI for this hour
                aqi = calculate_aqi(
                    pm25=pm2_5,
                    pm10=pm10,
                    o3=ozone,
                    no2=nitrogen_dioxide,
                    so2=sulphur_dioxide,
                    co=carbon_monoxide
                )
                
                # Determine trend (compare with previous hour) with percentage change
                trend = None
                trend_percentage = None
                if previous_aqi is not None and aqi is not None and previous_aqi > 0:
                    percentage_change = abs((aqi - previous_aqi) / previous_aqi) * 100
                    if aqi > previous_aqi:
                        trend = "↑"
                        trend_percentage = round(percentage_change, 1)
                    elif aqi < previous_aqi:
                        trend = "↓"
                        trend_percentage = round(percentage_change, 1)
                    else:
                        trend = "→"
                        trend_percentage = 0.0
                
                # Determine data source based on current time
                if is_today:
                    # Compare hour timestamp with current time
                    # If the hour has passed (timestamp < current time), it's historical
                    if hour_timestamp < current_time_timestamp:
                        data_source = "History/Estimate"
                    # If the hour is in the future, it's prediction
                    else:
                        data_source = "Prediction"
                else:
                    # For past dates, everything is historical
                    data_source = "History/Estimate"
                
                # Calculate correction factor for this specific hour
                # Pass the datetime object directly to ensure exact matching
                hour_datetime_str = hour_datetime.strftime('%Y-%m-%d %H:%M:%S')
                cf_data = calculate_correction_factor(latitude, longitude, hour_datetime)
                cf = cf_data["cf"]
                
                # Calculate corrected values
                pm10_0to3m = round(pm10 * cf, 2) if pm10 is not None else None
                pm2_5_0to3m = round(pm2_5 * cf, 2) if pm2_5 is not None else None
                co_0to3m = round(carbon_monoxide * cf, 2) if carbon_monoxide is not None else None
                no2_0to3m = round(nitrogen_dioxide * cf, 2) if nitrogen_dioxide is not None else None
                so2_0to3m = round(sulphur_dioxide * cf, 2) if sulphur_dioxide is not None else None
                o3_0to3m = round(ozone * cf, 2) if ozone is not None else None
                
                aqi_corrected = calculate_aqi(
                    pm25=pm2_5_0to3m,
                    pm10=pm10_0to3m,
                    o3=o3_0to3m,
                    no2=no2_0to3m,
                    so2=so2_0to3m,
                    co=co_0to3m
                )
                aqi_corrected = int(aqi_corrected) if aqi_corrected is not None else None

                # Calculate corrected AQI by multiplying raw AQI with correction factor
                aqi_0to3m = round(aqi * cf, 2) if aqi is not None else None
                
                record = {
                    "date": hour_datetime_str,
                    "pm10": round(pm10, 2) if pm10 is not None else None,
                    "pm2_5": round(pm2_5, 2) if pm2_5 is not None else None,
                    "co": round(carbon_monoxide, 2) if carbon_monoxide is not None else None,
                    "no2": round(nitrogen_dioxide, 2) if nitrogen_dioxide is not None else None,
                    "so2": round(sulphur_dioxide, 2) if sulphur_dioxide is not None else None,
                    "o3": round(ozone, 2) if ozone is not None else None,
                    "aqi": int(aqi) if aqi is not None else None,
                    "trend": trend,
                    "trend_percentage": trend_percentage,
                    "data_source": data_source,
                    # Corrected values
                    "aqi_corrected": aqi_corrected,
                    "aqi_0to3m": int(aqi_0to3m) if aqi_0to3m is not None else None,
                    "pm10_0to3m": pm10_0to3m,
                    "pm2_5_0to3m": pm2_5_0to3m,
                    "co_0to3m": co_0to3m,
                    "no2_0to3m": no2_0to3m,
                    "so2_0to3m": so2_0to3m,
                    "o3_0to3m": o3_0to3m,
                    # Correction factor details
                    "correction_factor": {
                        "cf": cf,
                        "vpc": cf_data["vpc"],
                        "tdc": cf_data["tdc"],
                        "lsaf": cf_data["lsaf"],
                        "hcf": cf_data["hcf"],
                        "rh": cf_data["rh"]
                    }
                }
                
                hourly_records.append(record)
                previous_aqi = aqi
                
            except (IndexError, TypeError, ValueError) as e:
                logger.error(f"Error processing hour {i}: {str(e)}")
                continue
        
        return jsonify({
            "date": date,
            "hourly_records": hourly_records
        }), 200
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error fetching hourly AQI data: {str(e)}")
        logger.error(f"Traceback: {error_trace}")
        return jsonify({
            'error': f'Failed to fetch hourly AQI data: {str(e)}',
            'details': 'Please check the backend logs for more information'
        }), 500


@app.route('/api/aqi/hourly/range', methods=['POST'])
def get_aqi_hourly_range():
    """
    Get hourly Air Quality Index data for a date range
    Expects: { "latitude": float, "longitude": float, "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD" }
    Returns: Array of hourly AQI records for all dates in the range
    """
    try:
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not latitude or not longitude:
            return jsonify({'error': 'Latitude and longitude are required'}), 400
        
        if not start_date or not end_date:
            return jsonify({'error': 'Start date and end date are required'}), 400
        
        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        
        # Fetch data for each date in the range
        all_hourly_records = []
        current_date = start_date_obj
        
        while current_date <= end_date_obj:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Use the existing hourly endpoint logic
            url = "https://air-quality-api.open-meteo.com/v1/air-quality"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", 
                          "sulphur_dioxide", "ozone"],
                "start_date": date_str,
                "end_date": date_str,
            }
            
            # For today, request tomorrow too to get full forecast
            today = datetime.now().date()
            if current_date.date() == today:
                tomorrow = (current_date.date() + timedelta(days=1)).strftime('%Y-%m-%d')
                params["end_date"] = tomorrow
            
            try:
                responses = openmeteo.weather_api(url, params=params)
                if responses:
                    response = responses[0]
                    hourly = response.Hourly()
                    hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
                    hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()
                    hourly_carbon_monoxide = hourly.Variables(2).ValuesAsNumpy()
                    hourly_nitrogen_dioxide = hourly.Variables(3).ValuesAsNumpy()
                    hourly_sulphur_dioxide = hourly.Variables(4).ValuesAsNumpy()
                    hourly_ozone = hourly.Variables(5).ValuesAsNumpy()
                    
                    # Get hourly times
                    try:
                        start_time = float(hourly.Time())
                        num_values = len(hourly_pm10)
                        try:
                            interval = float(hourly.Interval())
                        except (AttributeError, TypeError, ValueError):
                            interval = 3600
                        hourly_times = [start_time + (i * interval) for i in range(num_values)]
                    except (AttributeError, TypeError, ValueError):
                        try:
                            hourly_times_array = hourly.Time()
                            if hasattr(hourly_times_array, '__iter__') and not isinstance(hourly_times_array, (str, bytes)):
                                hourly_times = list(hourly_times_array)
                            else:
                                hourly_times = []
                        except Exception:
                            hourly_times = []
                    
                    # Process hours for this date
                    now = datetime.now(timezone.utc)
                    current_time_timestamp = now.timestamp()
                    requested_date_obj = current_date.date()
                    requested_date_start = datetime.combine(requested_date_obj, datetime.min.time()).replace(tzinfo=timezone.utc)
                    requested_date_end = requested_date_start + timedelta(days=1)
                    is_today = requested_date_obj == today
                    
                    def safe_float(value):
                        if value is None:
                            return None
                        try:
                            val = float(value)
                            if val != val:
                                return None
                            return val
                        except (TypeError, ValueError):
                            return None
                    
                    for i in range(len(hourly_pm10)):
                        try:
                            hour_timestamp = float(hourly_times[i]) if i < len(hourly_times) else None
                            if hour_timestamp:
                                hour_datetime = datetime.fromtimestamp(hour_timestamp, tz=timezone.utc)
                            else:
                                continue
                            
                            if hour_datetime < requested_date_start or hour_datetime >= requested_date_end:
                                continue
                            
                            pm10 = safe_float(hourly_pm10[i])
                            pm2_5 = safe_float(hourly_pm2_5[i])
                            carbon_monoxide = safe_float(hourly_carbon_monoxide[i])
                            nitrogen_dioxide = safe_float(hourly_nitrogen_dioxide[i])
                            sulphur_dioxide = safe_float(hourly_sulphur_dioxide[i])
                            ozone = safe_float(hourly_ozone[i])
                            
                            aqi = calculate_aqi(
                                pm25=pm2_5,
                                pm10=pm10,
                                o3=ozone,
                                no2=nitrogen_dioxide,
                                so2=sulphur_dioxide,
                                co=carbon_monoxide
                            )
                            
                            data_source = "History/Estimate"
                            if is_today:
                                if hour_timestamp < current_time_timestamp:
                                    data_source = "History/Estimate"
                                else:
                                    data_source = "Prediction"
                            
                            # Calculate correction factor for this specific hour
                            # Pass the datetime object directly to ensure exact matching
                            hour_datetime_str = hour_datetime.strftime('%Y-%m-%d %H:%M:%S')
                            cf_data = calculate_correction_factor(latitude, longitude, hour_datetime)
                            cf = cf_data["cf"]
                            
                            # Calculate corrected values
                            pm10_0to3m = round(pm10 * cf, 2) if pm10 is not None else None
                            pm2_5_0to3m = round(pm2_5 * cf, 2) if pm2_5 is not None else None
                            co_0to3m = round(carbon_monoxide * cf, 2) if carbon_monoxide is not None else None
                            no2_0to3m = round(nitrogen_dioxide * cf, 2) if nitrogen_dioxide is not None else None
                            so2_0to3m = round(sulphur_dioxide * cf, 2) if sulphur_dioxide is not None else None
                            o3_0to3m = round(ozone * cf, 2) if ozone is not None else None
                            aqi_corrected = calculate_aqi(
                                pm25=pm2_5_0to3m,
                                pm10=pm10_0to3m,
                                o3=o3_0to3m,
                                no2=no2_0to3m,
                                so2=so2_0to3m,
                                co=co_0to3m
                            )
                            aqi_corrected = int(aqi_corrected) if aqi_corrected is not None else None

                            # Calculate corrected AQI by multiplying raw AQI with correction factor
                            aqi_0to3m = round(aqi * cf, 2) if aqi is not None else None
                            
                            record = {
                                "date": hour_datetime_str,
                                "pm10": round(pm10, 2) if pm10 is not None else None,
                                "pm2_5": round(pm2_5, 2) if pm2_5 is not None else None,
                                "co": round(carbon_monoxide, 2) if carbon_monoxide is not None else None,
                                "no2": round(nitrogen_dioxide, 2) if nitrogen_dioxide is not None else None,
                                "so2": round(sulphur_dioxide, 2) if sulphur_dioxide is not None else None,
                                "o3": round(ozone, 2) if ozone is not None else None,
                                "aqi": int(aqi) if aqi is not None else None,
                                "data_source": data_source,
                                # Corrected values
                                "aqi_corrected": aqi_corrected,
                                "aqi_0to3m": int(aqi_0to3m) if aqi_0to3m is not None else None,
                                "pm10_0to3m": pm10_0to3m,
                                "pm2_5_0to3m": pm2_5_0to3m,
                                "co_0to3m": co_0to3m,
                                "no2_0to3m": no2_0to3m,
                                "so2_0to3m": so2_0to3m,
                                "o3_0to3m": o3_0to3m,
                                # Correction factor details
                                "correction_factor": {
                                    "cf": cf,
                                    "vpc": cf_data["vpc"],
                                    "tdc": cf_data["tdc"],
                                    "lsaf": cf_data["lsaf"],
                                    "hcf": cf_data["hcf"],
                                    "rh": cf_data["rh"]
                                }
                            }
                            
                            all_hourly_records.append(record)
                        except Exception:
                            continue
            except Exception as e:
                logger.error(f"Error fetching data for {date_str}: {str(e)}")
                continue
            
            current_date += timedelta(days=1)
        
        return jsonify({
            "start_date": start_date,
            "end_date": end_date,
            "hourly_records": all_hourly_records
        }), 200
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error fetching hourly AQI data range: {str(e)}")
        logger.error(f"Traceback: {error_trace}")
        return jsonify({
            'error': f'Failed to fetch hourly AQI data range: {str(e)}',
            'details': 'Please check the backend logs for more information'
        }), 500


@app.route('/api/weather', methods=['POST'])
def get_weather():
    """
    Get Weather data from Open-Meteo
    Expects: { "latitude": float, "longitude": float, "date": "YYYY-MM-DD" }
    """
    try:
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        date = data.get('date')  # Optional: specific date
        
        if not latitude or not longitude:
            return jsonify({'error': 'Latitude and longitude are required'}), 400
        
        # Use forecast API for current weather with UV index
        url = "https://api.open-meteo.com/v1/forecast"
        
        # Check if this is a historical date (before today)
        today = datetime.now().date()
        is_historical = False
        if date:
            try:
                request_date = datetime.strptime(date, '%Y-%m-%d').date()
                is_historical = request_date < today
            except ValueError:
                is_historical = False
        
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": "auto"
        }
        
        if is_historical:
            # For historical dates, use hourly data to calculate daily averages
            params["hourly"] = ["temperature_2m", "relative_humidity_2m", "apparent_temperature", 
                               "weather_code", "wind_speed_10m", "uv_index"]
            params["daily"] = ["temperature_2m_max", "temperature_2m_min", "weather_code", 
                             "wind_speed_10m_max", "relative_humidity_2m_max", "uv_index_max"]
        else:
            # For current date, use hourly and daily data
            params["hourly"] = ["temperature_2m", "relative_humidity_2m", "apparent_temperature", 
                               "weather_code", "wind_speed_10m", "uv_index"]
            params["daily"] = ["temperature_2m_max", "temperature_2m_min", "weather_code", 
                             "wind_speed_10m_max", "relative_humidity_2m_max", "uv_index_max"]
        
        if date:
            try:
                date_obj = datetime.strptime(date, '%Y-%m-%d')
                params["start_date"] = date_obj.strftime('%Y-%m-%d')
                params["end_date"] = date_obj.strftime('%Y-%m-%d')
            except ValueError:
                return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        
        responses = openmeteo.weather_api(url, params=params)
        
        if not responses:
            return jsonify({'error': 'No data available for the specified location'}), 404
        
        response = responses[0]
        
        # Process daily data for the specific date
        daily = response.Daily()
        daily_temperature_max = daily.Variables(0).ValuesAsNumpy()
        daily_temperature_min = daily.Variables(1).ValuesAsNumpy()
        daily_weather_code = daily.Variables(2).ValuesAsNumpy()
        daily_wind_speed_max = daily.Variables(3).ValuesAsNumpy()
        daily_humidity_max = daily.Variables(4).ValuesAsNumpy()
        daily_uv_index_max = daily.Variables(5).ValuesAsNumpy()
        
        # Get weather condition from WMO code
        weather_conditions = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
            55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Light freezing rain",
            67: "Heavy freezing rain", 71: "Slight snow fall", 73: "Moderate snow fall",
            75: "Heavy snow fall", 77: "Snow grains", 80: "Slight rain showers",
            81: "Moderate rain showers", 82: "Violent rain showers", 85: "Slight snow showers",
            86: "Heavy snow showers", 95: "Thunderstorm", 96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }
        
        if is_historical:
            # For historical dates, calculate daily averages from hourly data
            hourly = response.Hourly()
            hourly_temperature = hourly.Variables(0).ValuesAsNumpy()
            hourly_humidity = hourly.Variables(1).ValuesAsNumpy()
            hourly_apparent_temperature = hourly.Variables(2).ValuesAsNumpy()
            hourly_weather_code = hourly.Variables(3).ValuesAsNumpy()
            hourly_wind_speed = hourly.Variables(4).ValuesAsNumpy()
            hourly_uv_index = hourly.Variables(5).ValuesAsNumpy()
            
            # Calculate daily averages (mean of all hourly values)
            avg_temperature = float(hourly_temperature.mean()) if len(hourly_temperature) > 0 else None
            avg_humidity = float(hourly_humidity.mean()) if len(hourly_humidity) > 0 else None
            avg_apparent_temperature = float(hourly_apparent_temperature.mean()) if len(hourly_apparent_temperature) > 0 else None
            avg_wind_speed = float(hourly_wind_speed.mean()) if len(hourly_wind_speed) > 0 else None
            avg_uv_index = float(hourly_uv_index.mean()) if len(hourly_uv_index) > 0 else None
            
            # Get most common weather code (mode)
            if len(hourly_weather_code) > 0:
                from collections import Counter
                weather_code_counts = Counter(hourly_weather_code.astype(int))
                most_common_weather_code = int(weather_code_counts.most_common(1)[0][0])
            else:
                most_common_weather_code = int(float(daily_weather_code[0])) if len(daily_weather_code) > 0 else 0
            
            weather_condition = weather_conditions.get(most_common_weather_code, "Unknown")
            
            # Get uv_index_max from daily data
            daily_uv_max = round(float(daily_uv_index_max[0]), 1) if len(daily_uv_index_max) > 0 else None
            
            result = {
                "date": date + " 12:00:00",
                "temperature": round(avg_temperature, 1) if avg_temperature is not None else None,
                "feels_like": round(avg_apparent_temperature, 1) if avg_apparent_temperature is not None else None,
                "humidity": round(avg_humidity, 1) if avg_humidity is not None else None,
                "wind_speed": round(avg_wind_speed, 1) if avg_wind_speed is not None else None,
                "uv_index": round(avg_uv_index, 1) if avg_uv_index is not None else None,
                "uv_index_max": daily_uv_max,
                "condition": weather_condition,
                "weather_code": int(most_common_weather_code),
                "icon": "cloud" if most_common_weather_code >= 2 else "sun",
                "daily": {
                    "temp_max": round(float(daily_temperature_max[0]), 1) if len(daily_temperature_max) > 0 else None,
                    "temp_min": round(float(daily_temperature_min[0]), 1) if len(daily_temperature_min) > 0 else None,
                    "wind_speed_max": round(float(daily_wind_speed_max[0]), 1) if len(daily_wind_speed_max) > 0 else None,
                    "humidity_max": round(float(daily_humidity_max[0]), 1) if len(daily_humidity_max) > 0 else None,
                    "uv_index_max": daily_uv_max,
                }
            }
        else:
            # For current date, use hourly data and get the most recent complete hour
            # Initialize hour_datetime to current time as fallback
            now_init = datetime.now(timezone.utc)
            current_hour_init = now_init.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            hour_datetime = current_hour_init
            
            hourly = response.Hourly()
            hourly_temperature = hourly.Variables(0).ValuesAsNumpy()
            hourly_humidity = hourly.Variables(1).ValuesAsNumpy()
            hourly_apparent_temperature = hourly.Variables(2).ValuesAsNumpy()
            hourly_weather_code = hourly.Variables(3).ValuesAsNumpy()
            hourly_wind_speed = hourly.Variables(4).ValuesAsNumpy()
            hourly_uv_index = hourly.Variables(5).ValuesAsNumpy()
            
            # Get hourly times - Open-Meteo returns Unix timestamps in seconds
            # hourly.Time() might return a single start timestamp or an array
            # We'll calculate timestamps from start time and interval if needed
            try:
                start_time = float(hourly.Time())
                num_values = len(hourly_temperature)
                
                # Try to get interval, default to 3600 seconds (1 hour) if not available
                try:
                    interval = float(hourly.Interval())  # Interval in seconds
                except (AttributeError, TypeError, ValueError):
                    interval = 3600  # Default to 1 hour
                
                # Calculate all timestamps from start time and interval
                hourly_times = [start_time + (i * interval) for i in range(num_values)]
            except (AttributeError, TypeError, ValueError) as e:
                logger.error(f"Error getting hourly times: {str(e)}")
                # Fallback: try to get as array
                try:
                    hourly_times_array = hourly.Time()
                    if hasattr(hourly_times_array, '__iter__') and not isinstance(hourly_times_array, (str, bytes)):
                        hourly_times = list(hourly_times_array)
                    else:
                        hourly_times = []
                except Exception as e2:
                    logger.error(f"Error in fallback time parsing: {str(e2)}")
                    hourly_times = []
            
            # Get current time and find the most recent complete hour
            now = datetime.now(timezone.utc)
            # Get the most recent complete hour (subtract 1 hour from current hour)
            current_hour = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            current_hour_timestamp = current_hour.timestamp()
            
            # Find the index of the most recent complete hour (not current or future)
            latest_index = -1
            if len(hourly_times) > 0:
                # Iterate backwards to find the most recent hour <= current_hour
                for i in range(len(hourly_times) - 1, -1, -1):
                    try:
                        # hourly.Time() returns Unix timestamps in seconds
                        hour_timestamp = float(hourly_times[i])
                        hour_datetime = datetime.fromtimestamp(hour_timestamp, tz=timezone.utc)
                        
                        # Use the most recent hour that is <= current_hour (most recent complete hour)
                        if hour_timestamp <= current_hour_timestamp:
                            latest_index = i
                            break
                    except (IndexError, TypeError, ValueError, OSError) as e:
                        logger.error(f"Error parsing hour time at index {i}: {str(e)}")
                        continue
            
            # If no valid hour found, use the last available (simpler fallback)
            if latest_index == -1:
                if len(hourly_temperature) > 0:
                    latest_index = len(hourly_temperature) - 1
                else:
                    latest_index = -1
            
            # Get values for the most recent complete hour
            if latest_index >= 0 and latest_index < len(hourly_temperature):
                try:
                    def safe_float(value):
                        if value is None:
                            return None
                        try:
                            val = float(value)
                            # Check for NaN
                            if val != val:  # NaN check
                                return None
                            return val
                        except (TypeError, ValueError):
                            return None
                    
                    temperature = safe_float(hourly_temperature[latest_index])
                    humidity = safe_float(hourly_humidity[latest_index])
                    apparent_temperature = safe_float(hourly_apparent_temperature[latest_index])
                    weather_code_val = safe_float(hourly_weather_code[latest_index])
                    weather_code = int(weather_code_val) if weather_code_val is not None else 0
                    wind_speed = safe_float(hourly_wind_speed[latest_index])
                    uv_index = safe_float(hourly_uv_index[latest_index])
                except (IndexError, TypeError, ValueError) as e:
                    logger.error(f"Error accessing hourly weather data at index {latest_index}: {str(e)}")
                    temperature = None
                    humidity = None
                    apparent_temperature = None
                    weather_code = 0
                    wind_speed = None
                    uv_index = None
                
                # Get the timestamp for this hour
                try:
                    if latest_index < len(hourly_times):
                        # hourly.Time() returns Unix timestamps in seconds
                        hour_timestamp = float(hourly_times[latest_index])
                        hour_datetime = datetime.fromtimestamp(hour_timestamp, tz=timezone.utc)
                    else:
                        hour_datetime = current_hour
                except (IndexError, TypeError, ValueError, OSError):
                    hour_datetime = current_hour
            else:
                # Fallback if no data available
                temperature = None
                humidity = None
                apparent_temperature = None
                weather_code = 0
                wind_speed = None
                uv_index = None
                hour_datetime = current_hour
            
            weather_condition = weather_conditions.get(weather_code, "Unknown")
            
            # Get uv_index_max from daily data
            daily_uv_max = round(float(daily_uv_index_max[0]), 1) if len(daily_uv_index_max) > 0 else None
            
            result = {
                "date": hour_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                "temperature": round(temperature, 1) if temperature is not None else None,
                "feels_like": round(apparent_temperature, 1) if apparent_temperature is not None else None,
                "humidity": round(humidity, 1) if humidity is not None else None,
                "wind_speed": round(wind_speed, 1) if wind_speed is not None else None,
                "uv_index": round(uv_index, 1) if uv_index is not None else None,
                "uv_index_max": daily_uv_max,
                "condition": weather_condition,
                "weather_code": int(weather_code),
                "icon": "cloud" if weather_code >= 2 else "sun",
                "daily": {
                    "temp_max": round(float(daily_temperature_max[0]), 1) if len(daily_temperature_max) > 0 else None,
                    "temp_min": round(float(daily_temperature_min[0]), 1) if len(daily_temperature_min) > 0 else None,
                    "wind_speed_max": round(float(daily_wind_speed_max[0]), 1) if len(daily_wind_speed_max) > 0 else None,
                    "humidity_max": round(float(daily_humidity_max[0]), 1) if len(daily_humidity_max) > 0 else None,
                    "uv_index_max": daily_uv_max,
                }
            }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error fetching weather data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/weather/hourly', methods=['POST'])
def get_weather_hourly():
    """
    Get hourly Weather data from Open-Meteo for a specific date
    Expects: { "latitude": float, "longitude": float, "date": "YYYY-MM-DD" }
    Returns: Array of hourly weather records
    """
    try:
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        date = data.get('date')  # Required: specific date

        if not latitude or not longitude:
            return jsonify({'error': 'Latitude and longitude are required'}), 400

        if not date:
            return jsonify({'error': 'Date is required'}), 400

        try:
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            start_date = date_obj.strftime('%Y-%m-%d')
            end_date = date_obj.strftime('%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", 
                      "weather_code", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
                      "uv_index", "precipitation", "cloud_cover", "visibility", "surface_pressure"],
            "start_date": start_date,
            "end_date": end_date,
        }
        
        # For today, ensure we get forecast data by requesting up to tomorrow
        today = datetime.now().date()
        is_today = False
        try:
            request_date = datetime.strptime(date, '%Y-%m-%d').date()
            is_today = request_date == today
            if is_today:
                # Request data for today and tomorrow to get full forecast
                tomorrow = (request_date + timedelta(days=1)).strftime('%Y-%m-%d')
                params["end_date"] = tomorrow
        except ValueError:
            pass

        responses = openmeteo.weather_api(url, params=params)

        if not responses:
            return jsonify({'error': 'No data available for the specified location'}), 404

        response = responses[0]
        hourly = response.Hourly()
        hourly_temperature = hourly.Variables(0).ValuesAsNumpy()
        hourly_humidity = hourly.Variables(1).ValuesAsNumpy()
        hourly_apparent_temperature = hourly.Variables(2).ValuesAsNumpy()
        hourly_weather_code = hourly.Variables(3).ValuesAsNumpy()
        hourly_wind_speed = hourly.Variables(4).ValuesAsNumpy()
        hourly_wind_direction = hourly.Variables(5).ValuesAsNumpy()
        hourly_wind_gusts = hourly.Variables(6).ValuesAsNumpy()
        hourly_uv_index = hourly.Variables(7).ValuesAsNumpy()
        hourly_precipitation = hourly.Variables(8).ValuesAsNumpy()
        hourly_cloud_cover = hourly.Variables(9).ValuesAsNumpy()
        hourly_visibility = hourly.Variables(10).ValuesAsNumpy()
        try:
            hourly_pressure = hourly.Variables(11).ValuesAsNumpy()
        except (IndexError, AttributeError, TypeError):
            # If pressure is not available, create an array of None values
            hourly_pressure = [None] * len(hourly_temperature)

        # Get hourly times
        try:
            start_time = float(hourly.Time())
            num_values = len(hourly_temperature)
            try:
                interval = float(hourly.Interval())
            except (AttributeError, TypeError, ValueError):
                interval = 3600  # Default to 1 hour
            hourly_times = [start_time + (i * interval) for i in range(num_values)]
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error getting hourly times: {str(e)}")
            try:
                hourly_times_array = hourly.Time()
                if hasattr(hourly_times_array, '__iter__') and not isinstance(hourly_times_array, (str, bytes)):
                    hourly_times = list(hourly_times_array)
                else:
                    hourly_times = []
            except Exception as e2:
                logger.error(f"Error in fallback time parsing: {str(e2)}")
                hourly_times = []

        # Get current time for data source labeling
        now = datetime.now(timezone.utc)
        current_time_timestamp = now.timestamp()

        # Weather condition mapping
        weather_conditions = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
            55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Light freezing rain",
            67: "Heavy freezing rain", 71: "Slight snow fall", 73: "Moderate snow fall",
            75: "Heavy snow fall", 77: "Snow grains", 80: "Slight rain showers",
            81: "Moderate rain showers", 82: "Violent rain showers", 85: "Slight snow showers",
            86: "Heavy snow showers", 95: "Thunderstorm", 96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }

        def safe_float(value):
            if value is None:
                return None
            try:
                val = float(value)
                if val != val:  # NaN check
                    return None
                return val
            except (TypeError, ValueError):
                return None

        def get_wind_direction(degrees):
            """Convert wind direction degrees to cardinal direction"""
            if degrees is None:
                return None
            directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                         "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
            index = int((degrees + 11.25) / 22.5) % 16
            return directions[index]

        # Build hourly records
        hourly_records = []
        requested_date_obj = datetime.strptime(date, '%Y-%m-%d').date()
        requested_date_start = datetime.combine(requested_date_obj, datetime.min.time()).replace(tzinfo=timezone.utc)
        requested_date_end = requested_date_start + timedelta(days=1)

        for i in range(len(hourly_temperature)):
            try:
                hour_timestamp = float(hourly_times[i]) if i < len(hourly_times) else None
                if hour_timestamp:
                    hour_datetime = datetime.fromtimestamp(hour_timestamp, tz=timezone.utc)
                else:
                    continue

                # Only include hours that belong to the requested date
                if hour_datetime < requested_date_start or hour_datetime >= requested_date_end:
                    continue

                temperature = safe_float(hourly_temperature[i])
                humidity = safe_float(hourly_humidity[i])
                feels_like = safe_float(hourly_apparent_temperature[i])
                weather_code = int(hourly_weather_code[i]) if hourly_weather_code[i] is not None else None
                wind_speed = safe_float(hourly_wind_speed[i])
                wind_direction_deg = safe_float(hourly_wind_direction[i])
                wind_gusts = safe_float(hourly_wind_gusts[i])
                uv_index = safe_float(hourly_uv_index[i])
                precipitation = safe_float(hourly_precipitation[i])
                cloud_cover = safe_float(hourly_cloud_cover[i])
                visibility = safe_float(hourly_visibility[i])
                pressure = safe_float(hourly_pressure[i])

                # Determine data source
                if is_today:
                    if hour_timestamp < current_time_timestamp:
                        data_source = "History/Estimate"
                    else:
                        data_source = "Prediction"
                else:
                    data_source = "History/Estimate"

                record = {
                    "date": hour_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    "temperature": round(temperature, 1) if temperature is not None else None,
                    "feels_like": round(feels_like, 1) if feels_like is not None else None,
                    "humidity": round(humidity, 1) if humidity is not None else None,
                    "wind_speed": round(wind_speed, 1) if wind_speed is not None else None,
                    "wind_direction": wind_direction_deg,
                    "wind_direction_cardinal": get_wind_direction(wind_direction_deg),
                    "wind_gusts": round(wind_gusts, 1) if wind_gusts is not None else None,
                    "uv_index": round(uv_index, 1) if uv_index is not None else None,
                    "precipitation": round(precipitation, 2) if precipitation is not None else None,
                    "cloud_cover": round(cloud_cover, 1) if cloud_cover is not None else None,
                    "visibility": round(visibility / 1000, 1) if visibility is not None else None,  # Convert m to km
                    "pressure": round(pressure, 1) if pressure is not None else None,  # Surface pressure in hPa (mb)
                    "weather_code": weather_code,
                    "condition": weather_conditions.get(weather_code, "Unknown") if weather_code is not None else None,
                    "data_source": data_source
                }

                hourly_records.append(record)

            except (IndexError, TypeError, ValueError) as e:
                logger.error(f"Error processing hour {i}: {str(e)}")
                continue

        return jsonify({
            "date": date,
            "hourly_records": hourly_records
        }), 200

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error fetching hourly weather data: {str(e)}")
        logger.error(f"Traceback: {error_trace}")
        return jsonify({
            'error': f'Failed to fetch hourly weather data: {str(e)}',
            'details': 'Please check the backend logs for more information'
        }), 500


@app.route('/api/weather/monthly', methods=['POST'])
def get_weather_monthly():
    """
    Get monthly weather forecast data from Open-Meteo
    Expects: { "latitude": float, "longitude": float, "year": int, "month": int }
    Returns: Array of daily weather records for the month
    """
    try:
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        year = data.get('year')
        month = data.get('month')
        
        if not latitude or not longitude:
            return jsonify({'error': 'Latitude and longitude are required'}), 400
        
        if not year or not month:
            # Default to current month
            today = datetime.now().date()
            year = today.year
            month = today.month
        
        # Calculate first and last day of month
        if month == 12:
            first_day = datetime(year, month, 1).date()
            last_day = datetime(year + 1, 1, 1).date() - timedelta(days=1)
        else:
            first_day = datetime(year, month, 1).date()
            last_day = datetime(year, month + 1, 1).date() - timedelta(days=1)
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": ["temperature_2m_max", "temperature_2m_min", "weather_code", "precipitation_sum"],
            "start_date": first_day.strftime('%Y-%m-%d'),
            "end_date": last_day.strftime('%Y-%m-%d'),
            "timezone": "auto"
        }
        
        responses = openmeteo.weather_api(url, params=params)
        
        if not responses:
            return jsonify({'error': 'No data available for the specified location'}), 404
        
        response = responses[0]
        daily = response.Daily()
        daily_temperature_max = daily.Variables(0).ValuesAsNumpy()
        daily_temperature_min = daily.Variables(1).ValuesAsNumpy()
        daily_weather_code = daily.Variables(2).ValuesAsNumpy()
        daily_precipitation = daily.Variables(3).ValuesAsNumpy()
        
        # Get daily times
        try:
            start_time = float(daily.Time())
            num_values = len(daily_temperature_max)
            try:
                interval = float(daily.Interval())
            except (AttributeError, TypeError, ValueError):
                interval = 86400  # Default to 1 day in seconds
            daily_times = [start_time + (i * interval) for i in range(num_values)]
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error getting daily times: {str(e)}")
            daily_times = []
        
        # Weather condition mapping
        weather_conditions = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
            55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Light freezing rain",
            67: "Heavy freezing rain", 71: "Slight snow fall", 73: "Moderate snow fall",
            75: "Heavy snow fall", 77: "Snow grains", 80: "Slight rain showers",
            81: "Moderate rain showers", 82: "Violent rain showers", 85: "Slight snow showers",
            86: "Heavy snow showers", 95: "Thunderstorm", 96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }
        
        def get_weather_icon(weather_code):
            """Get weather icon based on weather code"""
            if weather_code is None:
                return "cloud"
            if weather_code in [0, 1]:
                return "sun"
            elif weather_code in [2]:
                return "partly-cloudy"
            elif weather_code in [3, 45, 48]:
                return "cloud"
            elif weather_code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82]:
                return "rain"
            elif weather_code in [71, 73, 75, 77, 85, 86]:
                return "snow"
            elif weather_code in [95, 96, 99]:
                return "thunderstorm"
            return "cloud"
        
        def get_weather_category(weather_code):
            """Categorize weather for summary"""
            if weather_code is None:
                return "cloudy"
            if weather_code in [0, 1]:
                return "sunny"
            elif weather_code in [2]:
                return "partly-cloudy"
            elif weather_code in [3, 45, 48]:
                return "cloudy"
            elif weather_code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99]:
                return "rainy"
            elif weather_code in [71, 73, 75, 77, 85, 86]:
                return "snowy"
            return "cloudy"
        
        # Build daily records
        daily_records = []
        for i in range(len(daily_temperature_max)):
            try:
                day_timestamp = float(daily_times[i]) if i < len(daily_times) else None
                if day_timestamp:
                    day_datetime = datetime.fromtimestamp(day_timestamp, tz=timezone.utc)
                else:
                    continue
                
                weather_code = int(daily_weather_code[i]) if daily_weather_code[i] is not None else None
                temp_max = float(daily_temperature_max[i]) if daily_temperature_max[i] is not None else None
                temp_min = float(daily_temperature_min[i]) if daily_temperature_min[i] is not None else None
                precipitation = float(daily_precipitation[i]) if daily_precipitation[i] is not None else None
                
                record = {
                    "date": day_datetime.strftime('%Y-%m-%d'),
                    "day": day_datetime.day,
                    "day_of_week": day_datetime.strftime('%a'),
                    "temperature_max": round(temp_max, 1) if temp_max is not None else None,
                    "temperature_min": round(temp_min, 1) if temp_min is not None else None,
                    "weather_code": weather_code,
                    "condition": weather_conditions.get(weather_code, "Unknown") if weather_code is not None else None,
                    "icon": get_weather_icon(weather_code),
                    "category": get_weather_category(weather_code),
                    "precipitation": round(precipitation, 2) if precipitation is not None else None
                }
                
                daily_records.append(record)
            except (IndexError, TypeError, ValueError) as e:
                logger.error(f"Error processing day {i}: {str(e)}")
                continue
        
        # Calculate summary statistics
        category_counts = {}
        for record in daily_records:
            category = record.get('category', 'cloudy')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return jsonify({
            "year": year,
            "month": month,
            "daily_records": daily_records,
            "summary": {
                "sunny": category_counts.get("sunny", 0),
                "partly-cloudy": category_counts.get("partly-cloudy", 0),
                "cloudy": category_counts.get("cloudy", 0),
                "rainy": category_counts.get("rainy", 0),
                "snowy": category_counts.get("snowy", 0)
            }
        }), 200
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error fetching monthly weather data: {str(e)}")
        logger.error(f"Traceback: {error_trace}")
        return jsonify({
            'error': f'Failed to fetch monthly weather data: {str(e)}',
            'details': 'Please check the backend logs for more information'
        }), 500

@app.route('/api/aqi/analyze', methods=['POST'])
def analyze_aqi_with_llm():
    """
    Combines HOURLY AQI trend + LLM explanation
    """
    try:
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        date = data.get('date')

        if not latitude or not longitude:
            return jsonify({'error': 'Latitude and longitude are required'}), 400

        # 🔹 Call HOURLY AQI endpoint internally
        with app.test_request_context(
            '/api/aqi/hourly',
            method='POST',
            json={
                "latitude": latitude,
                "longitude": longitude,
                "date": date
            }
        ):
            hourly_response = get_aqi_hourly()

        hourly_json = hourly_response[0].get_json()
        hourly_records = hourly_json.get("hourly_records", [])

        if not hourly_records:
            return jsonify({'error': 'No hourly AQI data available'}), 404

        # 🔹 Determine current hour
        current_hour = get_current_hour_record(hourly_records)
        aqi_value = current_hour.get("aqi") if current_hour else None
        category = get_aqi_category(aqi_value) if aqi_value is not None else "Unknown"

        # 🔹 Determine next hour (prediction)
        next_hour = get_next_hour_record(hourly_records, current_hour)
        
        # 🔹 If AQI is good, skip LLM for both analyses
        if aqi_value is None or aqi_value <= 50:
            return jsonify({
                "aqi_data": current_hour,
                "category": category,
                "analysis": "Air quality is good. No analysis required.",
                "analysis_0to3m": "Air quality is good. No analysis required."
            }), 200

        # 🔹 Build STRICT time-locked prompt for raw data
        prompt_raw = build_strict_time_prompt(
            current_hour=current_hour,
            hourly_records=hourly_records,
            next_hour=next_hour,
            use_0to3m=False
        )

        # 🔹 Build STRICT time-locked prompt for 0to3m data
        prompt_0to3m = build_strict_time_prompt(
            current_hour=current_hour,
            hourly_records=hourly_records,
            next_hour=next_hour,
            use_0to3m=True
        )

        # 🔹 Generate analysis for raw data
        llm_response_raw = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an air quality expert."},
                {"role": "user", "content": prompt_raw}
            ],
            temperature=0.2,
            max_tokens=800
        )
        analysis_text = llm_response_raw.choices[0].message.content

        # 🔹 Generate analysis for 0to3m data
        llm_response_0to3m = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an air quality expert."},
                {"role": "user", "content": prompt_0to3m}
            ],
            temperature=0.2,
            max_tokens=800
        )
        analysis_0to3m_text = llm_response_0to3m.choices[0].message.content

        return jsonify({
            "aqi_data": current_hour,        # current hour snapshot
            "category": category,
            "analysis": analysis_text,
            "analysis_0to3m": analysis_0to3m_text,
            "date": date
        }), 200

    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to analyze AQI",
            "details": str(e)
        }), 500

@app.route('/api/aqi/tiles', methods=['POST'])
def get_aqi_tiles():
    """
    Get tile-based AQI data for a polygon area (geometry-based)
    
    Accepts either:
    1. Polygon geometry (preferred):
    {
        "polygon": [
            [lat1, lon1],
            [lat2, lon2],
            [lat3, lon3],
            ...
        ],
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD",
        "tile_size_km": float (optional, default 0.5),
        "use_0to3m": bool (optional, default True)
    }
    
    Returns:
    {
        "tiles": [
            {
                "tile_id": str,
                "center": {"latitude": float, "longitude": float},
                "bounds": {"min_lat": float, "max_lat": float, "min_lon": float, "max_lon": float},
                "aqi": int,
                "aqi_0to3m": int,
                "color": str (hex),
                "category": str,
                "label": str,
                "pm2_5": float,
                "pm10": float,
                "pm2_5_0to3m": float,
                "pm10_0to3m": float
            }
        ],
        "summary": {
            "total_tiles": int,
            "avg_aqi": float,
            "avg_aqi_0to3m": float,
            "min_aqi": int,
            "max_aqi": int
        }
    }
    """
    try:
        data = request.get_json()
        polygon = data.get('polygon')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        tile_size_km = data.get('tile_size_km', 0.5)  # Default 500m tiles
        
        if not start_date or not end_date:
            return jsonify({'error': 'Start date and end date are required'}), 400
        
        if not polygon:
            return jsonify({'error': 'Polygon is required'}), 400
        
        if not isinstance(polygon, list) or len(polygon) < 3:
            return jsonify({'error': 'Polygon must be a list of at least 3 [latitude, longitude] pairs'}), 400

        # Validate polygon coordinates
        for point in polygon:
            if not isinstance(point, list) or len(point) != 2:
                return jsonify({'error': 'Each polygon point must be [latitude, longitude]'}), 400
            lat, lon = point
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                return jsonify({'error': 'Coordinates must be numbers'}), 400
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                return jsonify({'error': 'Invalid coordinates: lat must be -90 to 90, lon must be -180 to 180'}), 400
        
        # Generate tiles for polygon
        tiles = generate_tiles_for_polygon(polygon, tile_size_km)
        bounds_data = get_polygon_bounds(polygon)
        
        if len(tiles) > 1000:  # Limit to prevent excessive API calls
            return jsonify({
                'error': f'Too many tiles ({len(tiles)}). Please use a smaller area or larger tile size.',
                'suggestion': f'Try tile_size_km >= {tile_size_km * (len(tiles) / 1000):.2f}'
            }), 400
        
        # Calculate AQI for each tile
        tile_results = []
        aqi_values = []
        tile_index = 0  # Initialize tile index for dummy AQI mapping
        
        # Use date range endpoint to get hourly data for date range
        # For each tile, we'll calculate average AQI over the date range
        for tile in tiles:
            try:
                center_lat = tile['center']['latitude']
                center_lon = tile['center']['longitude']
                
                # Fetch hourly AQI data for the date range
                url = "https://air-quality-api.open-meteo.com/v1/air-quality"
                params = {
                    "latitude": center_lat,
                    "longitude": center_lon,
                    "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", 
                              "sulphur_dioxide", "ozone"],
                    "start_date": start_date,
                    "end_date": end_date,
                }
                
                # For today, request tomorrow too to get full forecast
                today = datetime.now().date()
                try:
                    request_start = datetime.strptime(start_date, '%Y-%m-%d').date()
                    request_end = datetime.strptime(end_date, '%Y-%m-%d').date()
                    if request_start <= today <= request_end:
                        tomorrow = (today + timedelta(days=1)).strftime('%Y-%m-%d')
                        params["end_date"] = tomorrow
                except ValueError:
                    pass
                
                responses = openmeteo.weather_api(url, params=params)
                
                if not responses:
                    # Skip tile if no data available
                    continue
                
                response = responses[0]
                hourly = response.Hourly()
                hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
                hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()
                hourly_carbon_monoxide = hourly.Variables(2).ValuesAsNumpy()
                hourly_nitrogen_dioxide = hourly.Variables(3).ValuesAsNumpy()
                hourly_sulphur_dioxide = hourly.Variables(4).ValuesAsNumpy()
                hourly_ozone = hourly.Variables(5).ValuesAsNumpy()
                
                # Get hourly times
                try:
                    start_time = float(hourly.Time())
                    num_values = len(hourly_pm10)
                    try:
                        interval = float(hourly.Interval())
                    except (AttributeError, TypeError, ValueError):
                        interval = 3600
                    hourly_times = [start_time + (i * interval) for i in range(num_values)]
                except (AttributeError, TypeError, ValueError):
                    try:
                        hourly_times_array = hourly.Time()
                        if hasattr(hourly_times_array, '__iter__') and not isinstance(hourly_times_array, (str, bytes)):
                            hourly_times = list(hourly_times_array)
                        else:
                            hourly_times = []
                    except Exception:
                        hourly_times = []
                
                # Filter to date range and calculate averages
                def safe_mean(arr):
                    if len(arr) == 0:
                        return None
                    try:
                        mean_val = float(arr.mean())
                        if mean_val != mean_val:  # NaN check
                            return None
                        return mean_val
                    except (ValueError, TypeError, AttributeError):
                        return None
                
                # Filter hours within date range
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
                start_datetime = datetime.combine(start_date_obj, datetime.min.time()).replace(tzinfo=timezone.utc)
                end_datetime = datetime.combine(end_date_obj, datetime.min.time()).replace(tzinfo=timezone.utc) + timedelta(days=1)
                
                valid_pm10 = []
                valid_pm2_5 = []
                valid_co = []
                valid_no2 = []
                valid_so2 = []
                valid_o3 = []
                
                for i in range(len(hourly_pm10)):
                    try:
                        hour_timestamp = float(hourly_times[i]) if i < len(hourly_times) else None
                        if hour_timestamp:
                            hour_datetime = datetime.fromtimestamp(hour_timestamp, tz=timezone.utc)
                            if start_datetime <= hour_datetime < end_datetime:
                                pm10_val = float(hourly_pm10[i]) if hourly_pm10[i] is not None else None
                                pm2_5_val = float(hourly_pm2_5[i]) if hourly_pm2_5[i] is not None else None
                                co_val = float(hourly_carbon_monoxide[i]) if hourly_carbon_monoxide[i] is not None else None
                                no2_val = float(hourly_nitrogen_dioxide[i]) if hourly_nitrogen_dioxide[i] is not None else None
                                so2_val = float(hourly_sulphur_dioxide[i]) if hourly_sulphur_dioxide[i] is not None else None
                                o3_val = float(hourly_ozone[i]) if hourly_ozone[i] is not None else None
                                
                                if pm10_val is not None and pm10_val == pm10_val:  # Not NaN
                                    valid_pm10.append(pm10_val)
                                if pm2_5_val is not None and pm2_5_val == pm2_5_val:
                                    valid_pm2_5.append(pm2_5_val)
                                if co_val is not None and co_val == co_val:
                                    valid_co.append(co_val)
                                if no2_val is not None and no2_val == no2_val:
                                    valid_no2.append(no2_val)
                                if so2_val is not None and so2_val == so2_val:
                                    valid_so2.append(so2_val)
                                if o3_val is not None and o3_val == o3_val:
                                    valid_o3.append(o3_val)
                    except (IndexError, TypeError, ValueError):
                        continue
                
                # Calculate averages
                def safe_mean(values):
                    if not values:
                        return None
                    try:
                        return sum(values) / len(values)
                    except (TypeError, ValueError):
                        return None
                
                avg_pm10 = safe_mean(valid_pm10)
                avg_pm2_5 = safe_mean(valid_pm2_5)
                avg_co = safe_mean(valid_co)
                avg_no2 = safe_mean(valid_no2)
                avg_so2 = safe_mean(valid_so2)
                avg_o3 = safe_mean(valid_o3)
                
                # Calculate AQI at 10m height
                # aqi = calculate_aqi(
                #     pm25=avg_pm2_5,
                #     pm10=avg_pm10,
                #     o3=avg_o3,
                #     no2=avg_no2,
                #     so2=avg_so2,
                #     co=avg_co
                # )
                
                # ================= DUMMY AQI MAPPING (VISUAL TESTING ONLY) =================

                DUMMY_AQI_GROUPS = [
                    25, 25, 25,25,              # Good
                    75, 75, 75, 75,             # Moderate
                    110, 110, 110, 110,         # Poor
                    155, 155, 155, 155,         # Unhealthy
                    190, 190, 190,190,          # Unhealthy
                    260, 260, 260, 260,         # Severe
                    310, 310 ,310,310           # Hazardous
                ]

                aqi = DUMMY_AQI_GROUPS[tile_index % len(DUMMY_AQI_GROUPS)]
                tile_index += 1
                # =================================================================


                if aqi is None:
                    continue
               
                display_aqi = int(aqi)

                # Get color for the display AQI
                color_info = get_aqi_color(display_aqi)

                tile_result = {
                    "tile_id": tile["tile_id"],
                    "center": tile["center"],
                    "bounds": tile["bounds"],
                    "aqi": int(aqi),                 # 10m AQI ONLY
                    "aqi_0to3m": None,               # disabled
                    "color": color_info["color"],
                    "category": color_info["category"],
                    "label": color_info["label"],
                    "pm2_5": round(avg_pm2_5, 2) if avg_pm2_5 is not None else None,
                    "pm10": round(avg_pm10, 2) if avg_pm10 is not None else None,
                    "pm2_5_0to3m": None,             # disabled
                    "pm10_0to3m": None,              # disabled
                    "correction_factor": None        # disabled
                }

                
                tile_results.append(tile_result)
                aqi_values.append(aqi)
                
            except Exception as e:
                logger.error(f"Error processing tile {tile.get('tile_id', 'unknown')}: {str(e)}")
                continue
     
        
        summary = {
            "total_tiles": len(tile_results),
            "avg_aqi": round(sum(aqi_values) / len(aqi_values), 2) if aqi_values else None,
            "min_aqi": int(min(aqi_values)) if aqi_values else None,
            "max_aqi": int(max(aqi_values)) if aqi_values else None
        }

        return jsonify({
            "tiles": tile_results,
            "summary": summary,
            "bounds": bounds_data,
            "start_date": start_date,
            "end_date": end_date,
            "tile_size_km": tile_size_km,
        }), 200
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error fetching tile-based AQI data: {str(e)}")
        logger.error(f"Traceback: {error_trace}")
        return jsonify({
            'error': f'Failed to fetch tile-based AQI data: {str(e)}',
            'details': 'Please check the backend logs for more information'
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)