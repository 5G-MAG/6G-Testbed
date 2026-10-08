"""
Open-Meteo Weather MCP Server for the 6G AI Traffic Testbed.

Wraps the free Open-Meteo API (no API key required) to provide
weather data tools via MCP. Supports current conditions, forecasts,
geocoding, and air quality.

Usage (stdio):
    python -m mcp_servers.weather_server

Usage (HTTP via bridge):
    python -m mcp_servers.http_bridge python -m mcp_servers.weather_server
"""

import json
import sys
import urllib.request
import urllib.parse
from typing import Any

# ---------------------------------------------------------------------------
# Open-Meteo API helpers
# ---------------------------------------------------------------------------

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def _http_get(url: str) -> dict:
    """Simple blocking HTTP GET returning parsed JSON."""
    req = urllib.request.Request(url, headers={"User-Agent": "6g-ai-testbed/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def geocode_location(name: str, count: int = 3) -> list[dict]:
    """Geocode a place name to coordinates using Open-Meteo Geocoding API."""
    params = urllib.parse.urlencode({"name": name, "count": count, "language": "en"})
    data = _http_get(f"{_GEOCODE_URL}?{params}")
    results = data.get("results", [])
    return [
        {
            "name": r.get("name"),
            "country": r.get("country"),
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
            "elevation": r.get("elevation"),
            "timezone": r.get("timezone"),
        }
        for r in results
    ]


def get_current_weather(latitude: float, longitude: float) -> dict:
    """Get current weather conditions for a coordinate pair."""
    params = urllib.parse.urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join([
            "temperature_2m", "relative_humidity_2m", "apparent_temperature",
            "precipitation", "rain", "snowfall", "cloud_cover",
            "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
            "uv_index", "visibility", "surface_pressure",
        ]),
        "timezone": "auto",
    })
    data = _http_get(f"{_WEATHER_URL}?{params}")
    current = data.get("current", {})
    units = data.get("current_units", {})
    return {
        "location": {
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "timezone": data.get("timezone"),
            "elevation": data.get("elevation"),
        },
        "time": current.get("time"),
        "temperature_c": current.get("temperature_2m"),
        "feels_like_c": current.get("apparent_temperature"),
        "humidity_pct": current.get("relative_humidity_2m"),
        "precipitation_mm": current.get("precipitation"),
        "rain_mm": current.get("rain"),
        "snowfall_cm": current.get("snowfall"),
        "cloud_cover_pct": current.get("cloud_cover"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "wind_direction_deg": current.get("wind_direction_10m"),
        "wind_gusts_kmh": current.get("wind_gusts_10m"),
        "uv_index": current.get("uv_index"),
        "visibility_m": current.get("visibility"),
        "pressure_hpa": current.get("surface_pressure"),
        "units": units,
    }


def get_forecast(
    latitude: float,
    longitude: float,
    days: int = 3,
    hourly: bool = True,
) -> dict:
    """Get weather forecast for a coordinate pair."""
    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "forecast_days": min(days, 16),
        "timezone": "auto",
    }
    if hourly:
        params["hourly"] = ",".join([
            "temperature_2m", "precipitation_probability", "precipitation",
            "rain", "snowfall", "cloud_cover",
            "wind_speed_10m", "wind_gusts_10m", "uv_index", "visibility",
        ])
    else:
        params["daily"] = ",".join([
            "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "precipitation_probability_max",
            "wind_speed_10m_max", "wind_gusts_10m_max",
            "uv_index_max", "sunrise", "sunset",
        ])

    qs = urllib.parse.urlencode(params)
    data = _http_get(f"{_WEATHER_URL}?{qs}")

    result: dict[str, Any] = {
        "location": {
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "timezone": data.get("timezone"),
        },
    }
    if hourly and "hourly" in data:
        h = data["hourly"]
        times = h.get("time", [])
        result["hourly"] = [
            {
                "time": times[i],
                "temperature_c": h.get("temperature_2m", [None])[i] if i < len(h.get("temperature_2m", [])) else None,
                "precip_prob_pct": h.get("precipitation_probability", [None])[i] if i < len(h.get("precipitation_probability", [])) else None,
                "precipitation_mm": h.get("precipitation", [None])[i] if i < len(h.get("precipitation", [])) else None,
                "wind_speed_kmh": h.get("wind_speed_10m", [None])[i] if i < len(h.get("wind_speed_10m", [])) else None,
                "wind_gusts_kmh": h.get("wind_gusts_10m", [None])[i] if i < len(h.get("wind_gusts_10m", [])) else None,
                "uv_index": h.get("uv_index", [None])[i] if i < len(h.get("uv_index", [])) else None,
            }
            for i in range(len(times))
        ]
        result["hourly_units"] = data.get("hourly_units", {})
    elif "daily" in data:
        d = data["daily"]
        times = d.get("time", [])
        result["daily"] = [
            {
                "date": times[i],
                "temp_max_c": d.get("temperature_2m_max", [None])[i] if i < len(d.get("temperature_2m_max", [])) else None,
                "temp_min_c": d.get("temperature_2m_min", [None])[i] if i < len(d.get("temperature_2m_min", [])) else None,
                "precip_sum_mm": d.get("precipitation_sum", [None])[i] if i < len(d.get("precipitation_sum", [])) else None,
                "precip_prob_max_pct": d.get("precipitation_probability_max", [None])[i] if i < len(d.get("precipitation_probability_max", [])) else None,
                "wind_max_kmh": d.get("wind_speed_10m_max", [None])[i] if i < len(d.get("wind_speed_10m_max", [])) else None,
                "uv_max": d.get("uv_index_max", [None])[i] if i < len(d.get("uv_index_max", [])) else None,
            }
            for i in range(len(times))
        ]
        result["daily_units"] = data.get("daily_units", {})

    return result


def get_air_quality(latitude: float, longitude: float) -> dict:
    """Get current air quality for a coordinate pair."""
    params = urllib.parse.urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join([
            "european_aqi", "us_aqi",
            "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
            "sulphur_dioxide", "ozone",
        ]),
        "timezone": "auto",
    })
    data = _http_get(f"{_AIR_QUALITY_URL}?{params}")
    current = data.get("current", {})
    return {
        "location": {
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
        },
        "time": current.get("time"),
        "european_aqi": current.get("european_aqi"),
        "us_aqi": current.get("us_aqi"),
        "pm10_ugm3": current.get("pm10"),
        "pm2_5_ugm3": current.get("pm2_5"),
        "co_ugm3": current.get("carbon_monoxide"),
        "no2_ugm3": current.get("nitrogen_dioxide"),
        "so2_ugm3": current.get("sulphur_dioxide"),
        "ozone_ugm3": current.get("ozone"),
    }


# ---------------------------------------------------------------------------
# MCP stdio server (JSON-RPC over stdin/stdout)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "geocode_location",
        "description": "Convert a place name to latitude/longitude coordinates",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Place name to geocode"},
                "count": {"type": "integer", "description": "Max results (default 3)", "default": 3},
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_current_weather",
        "description": "Get current weather conditions for latitude/longitude",
        "inputSchema": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Latitude"},
                "longitude": {"type": "number", "description": "Longitude"},
            },
            "required": ["latitude", "longitude"],
        },
    },
    {
        "name": "get_forecast",
        "description": "Get weather forecast (hourly or daily) for a location",
        "inputSchema": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Latitude"},
                "longitude": {"type": "number", "description": "Longitude"},
                "days": {"type": "integer", "description": "Forecast days (1-16, default 3)", "default": 3},
                "hourly": {"type": "boolean", "description": "Hourly (true) or daily (false)", "default": True},
            },
            "required": ["latitude", "longitude"],
        },
    },
    {
        "name": "get_air_quality",
        "description": "Get current air quality index and pollutant levels",
        "inputSchema": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Latitude"},
                "longitude": {"type": "number", "description": "Longitude"},
            },
            "required": ["latitude", "longitude"],
        },
    },
]

_TOOL_DISPATCH = {
    "geocode_location": lambda args: geocode_location(args["name"], args.get("count", 3)),
    "get_current_weather": lambda args: get_current_weather(args["latitude"], args["longitude"]),
    "get_forecast": lambda args: get_forecast(
        args["latitude"], args["longitude"],
        args.get("days", 3), args.get("hourly", True),
    ),
    "get_air_quality": lambda args: get_air_quality(args["latitude"], args["longitude"]),
}


def _handle_request(request: dict) -> dict:
    """Handle a single JSON-RPC request."""
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "weather", "version": "1.0.0"},
            },
        }

    if method == "notifications/initialized":
        return None  # no response for notifications

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        handler = _TOOL_DISPATCH.get(tool_name)
        if not handler:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                    "isError": True,
                },
            }
        try:
            result = handler(arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {e}"}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main():
    """Run the MCP server over stdio."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = _handle_request(request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
