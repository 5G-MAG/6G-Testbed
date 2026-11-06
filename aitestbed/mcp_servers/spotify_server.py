#!/usr/bin/env python3
"""
Spotify MCP Server for the 6G AI Traffic Testbed.

Provides MCP tools for music search and playlist composition
via the Spotify Web API. Communicates over stdio using JSON-RPC 2.0.

Prerequisites:
    pip install requests
    export SPOTIFY_CLIENT_ID="your-client-id"
    export SPOTIFY_CLIENT_SECRET="your-client-secret"

    Get credentials at: https://developer.spotify.com/dashboard

Usage:
    python -m mcp_servers.spotify_server
"""

import json
import os
import sys
import time
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# Spotify Web API helpers
# ---------------------------------------------------------------------------


def _header_size(headers) -> int:
    """Estimate byte size of HTTP headers."""
    if not headers:
        return 0
    return sum(len(k) + len(v) + 4 for k, v in headers.items())  # "k: v\r\n"


class SpotifyAPI:
    """Thin wrapper around the Spotify Web API (client-credentials flow)."""

    TOKEN_URL = "https://accounts.spotify.com/api/token"
    API_BASE = "https://api.spotify.com/v1"

    def __init__(self):
        self.client_id = os.environ.get("SPOTIFY_CLIENT_ID", "")
        self.client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
        self._token: Optional[str] = None
        self._token_expires: float = 0.0
        self._last_backend_calls: list[dict] = []
        self._last_token_call: Optional[dict] = None

    def _ensure_token(self) -> str:
        """Obtain or refresh a client-credentials access token."""
        if self._token and time.time() < self._token_expires:
            return self._token

        if not self.client_id or not self.client_secret:
            raise RuntimeError(
                "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set"
            )

        t0 = time.time()
        resp = requests.post(
            self.TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
            timeout=10,
        )
        elapsed = time.time() - t0
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._token_expires = time.time() + data.get("expires_in", 3600) - 60

        self._last_token_call = {
            "url": self.TOKEN_URL,
            "http_status": resp.status_code,
            "request_bytes": len(resp.request.body or b"") + _header_size(resp.request.headers),
            "response_bytes": len(resp.content),
            "latency_sec": round(elapsed, 4),
        }

        return self._token

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._ensure_token()}"}

    MAX_RETRIES = 3
    RETRY_AFTER_CAP = 10  # respect Retry-After up to 10s to stay within client timeout

    def _get(self, url: str, params: dict, timeout: int = 15) -> requests.Response:
        """GET with automatic retry on 429 rate limit responses.

        Populates ``self._last_backend_calls`` with telemetry for each HTTP
        round-trip so the MCP response can include backend metrics.
        """
        calls: list[dict] = []
        for attempt in range(self.MAX_RETRIES):
            t0 = time.time()
            resp = requests.get(
                url, params=params, headers=self._headers(), timeout=timeout
            )
            elapsed = time.time() - t0

            call_info = {
                "url": resp.url,                    # full URL with query string
                "http_status": resp.status_code,
                "request_bytes": len(resp.request.url or "") + _header_size(resp.request.headers),
                "response_bytes": len(resp.content),
                "response_headers_bytes": _header_size(resp.headers),
                "latency_sec": round(elapsed, 4),
                "attempt": attempt + 1,
            }

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 1))
                call_info["rate_limited"] = True
                call_info["retry_after"] = retry_after
                calls.append(call_info)
                time.sleep(min(retry_after, self.RETRY_AFTER_CAP))
                continue

            calls.append(call_info)
            self._last_backend_calls = calls
            resp.raise_for_status()
            return resp

        calls.append(call_info)
        self._last_backend_calls = calls
        resp.raise_for_status()
        return resp

    # -- public helpers used by tools --

    def search_tracks(
        self, query: str, limit: int = 10, market: str = "US"
    ) -> list[dict]:
        resp = self._get(
            f"{self.API_BASE}/search",
            params={"q": query, "type": "track", "limit": limit, "market": market},
        )
        items = resp.json().get("tracks", {}).get("items", [])
        return [_track_summary(t) for t in items]

    def search_artists(
        self, query: str, limit: int = 5, market: str = "US"
    ) -> list[dict]:
        resp = self._get(
            f"{self.API_BASE}/search",
            params={"q": query, "type": "artist", "limit": limit, "market": market},
        )
        items = resp.json().get("artists", {}).get("items", [])
        return [_artist_summary(a) for a in items]

    def search_albums(
        self, query: str, limit: int = 5, market: str = "US"
    ) -> list[dict]:
        resp = self._get(
            f"{self.API_BASE}/search",
            params={"q": query, "type": "album", "limit": limit, "market": market},
        )
        items = resp.json().get("albums", {}).get("items", [])
        return [_album_summary(a) for a in items]

    def search_playlists(
        self, query: str, limit: int = 5, market: str = "US"
    ) -> list[dict]:
        resp = self._get(
            f"{self.API_BASE}/search",
            params={"q": query, "type": "playlist", "limit": limit, "market": market},
        )
        items = resp.json().get("playlists", {}).get("items", [])
        return [_playlist_summary(p) for p in items if p]

    def get_artist_albums(
        self, artist_id: str, limit: int = 10, market: str = "US"
    ) -> list[dict]:
        resp = self._get(
            f"{self.API_BASE}/artists/{artist_id}/albums",
            params={
                "include_groups": "album,single",
                "limit": limit,
                "market": market,
            },
        )
        items = resp.json().get("items", [])
        return [_album_summary(a) for a in items]

    def get_album_tracks(self, album_id: str, limit: int = 20) -> list[dict]:
        resp = self._get(
            f"{self.API_BASE}/albums/{album_id}/tracks",
            params={"limit": limit},
        )
        items = resp.json().get("items", [])
        return [_track_summary_simple(t) for t in items]


def _track_summary(track: dict) -> dict:
    artists = ", ".join(a["name"] for a in track.get("artists", []))
    album = track.get("album", {})
    return {
        "id": track.get("id"),
        "name": track.get("name"),
        "artists": artists,
        "album": album.get("name"),
        "release_date": album.get("release_date"),
        "duration_ms": track.get("duration_ms"),
        "popularity": track.get("popularity"),
        "preview_url": track.get("preview_url"),
        "spotify_url": track.get("external_urls", {}).get("spotify"),
    }


def _artist_summary(artist: dict) -> dict:
    return {
        "id": artist.get("id"),
        "name": artist.get("name"),
        "genres": artist.get("genres", []),
        "popularity": artist.get("popularity"),
        "followers": artist.get("followers", {}).get("total"),
        "spotify_url": artist.get("external_urls", {}).get("spotify"),
    }


def _album_summary(album: dict) -> dict:
    artists = ", ".join(a["name"] for a in album.get("artists", []))
    return {
        "id": album.get("id"),
        "name": album.get("name"),
        "artists": artists,
        "release_date": album.get("release_date"),
        "total_tracks": album.get("total_tracks"),
        "album_type": album.get("album_type"),
        "spotify_url": album.get("external_urls", {}).get("spotify"),
    }


def _playlist_summary(playlist: dict) -> dict:
    owner = playlist.get("owner", {})
    return {
        "id": playlist.get("id"),
        "name": playlist.get("name"),
        "description": playlist.get("description"),
        "owner": owner.get("display_name"),
        "total_tracks": playlist.get("tracks", {}).get("total"),
        "spotify_url": playlist.get("external_urls", {}).get("spotify"),
    }


def _track_summary_simple(track: dict) -> dict:
    """Summarize a track from the album-tracks endpoint (no album info)."""
    artists = ", ".join(a["name"] for a in track.get("artists", []))
    return {
        "id": track.get("id"),
        "name": track.get("name"),
        "artists": artists,
        "duration_ms": track.get("duration_ms"),
        "track_number": track.get("track_number"),
        "spotify_url": track.get("external_urls", {}).get("spotify"),
    }


# ---------------------------------------------------------------------------
# MCP tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "spotify_search_tracks",
        "description": "Search for tracks on Spotify by query string (song name, artist, etc.)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g. 'bohemian rhapsody', 'artist:Adele', 'genre:jazz')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (1-50, default 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "spotify_search_artists",
        "description": "Search for artists on Spotify.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Artist search query",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (1-50, default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "spotify_search_albums",
        "description": "Search for albums on Spotify.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Album search query",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (1-50, default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "spotify_search_playlists",
        "description": (
            "Search for playlists on Spotify by mood, genre, activity, or theme. "
            "Great for discovering curated collections (e.g. 'workout energy', 'chill jazz', 'road trip')."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Playlist search query (e.g. 'workout hits', 'relaxing piano', 'indie rock')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (1-50, default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "spotify_artist_albums",
        "description": "Get albums and singles for a specific artist by their Spotify artist ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "artist_id": {
                    "type": "string",
                    "description": "Spotify artist ID (from search results)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max albums to return (1-50, default 10)",
                    "default": 10,
                },
            },
            "required": ["artist_id"],
        },
    },
    {
        "name": "spotify_album_tracks",
        "description": "Get the track listing for a specific album by its Spotify album ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "album_id": {
                    "type": "string",
                    "description": "Spotify album ID (from search or artist albums results)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max tracks to return (1-50, default 20)",
                    "default": 20,
                },
            },
            "required": ["album_id"],
        },
    },
]


# ---------------------------------------------------------------------------
# MCP stdio server (JSON-RPC 2.0)
# ---------------------------------------------------------------------------

spotify = SpotifyAPI()


def handle_request(request: dict) -> dict:
    """Route a JSON-RPC request to the appropriate handler."""
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        try:
            from mcp.types import LATEST_PROTOCOL_VERSION
            protocol_version = LATEST_PROTOCOL_VERSION
        except ImportError:
            protocol_version = "2025-11-25"
        result = {
            "protocolVersion": protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "spotify-music",
                "version": "1.0.0",
            },
        }
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    if method == "notifications/initialized":
        # Notification — no id, no response expected.  Return None to
        # signal callers that nothing should be sent back.
        return None

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        return _handle_tool_call(req_id, params)

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


def _build_meta(tool_name: str, arguments: dict, error: str = None) -> dict:
    """Build telemetry metadata from the last Spotify API call(s)."""
    meta: dict = {
        "tool_name": tool_name,
        "tool_arguments": arguments,
    }

    if spotify._last_backend_calls:
        calls = spotify._last_backend_calls
        meta["backend_calls"] = calls
        meta["backend_total_request_bytes"] = sum(c.get("request_bytes", 0) for c in calls)
        meta["backend_total_response_bytes"] = sum(c.get("response_bytes", 0) for c in calls)
        meta["backend_total_latency_sec"] = round(sum(c.get("latency_sec", 0) for c in calls), 4)
        meta["backend_retries"] = len(calls) - 1
        meta["backend_rate_limited"] = any(c.get("rate_limited") for c in calls)
        spotify._last_backend_calls = []

    if spotify._last_token_call:
        meta["token_call"] = spotify._last_token_call
        spotify._last_token_call = None

    if error:
        meta["error"] = error

    return meta


def _handle_tool_call(req_id, params: dict) -> dict:
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    try:
        if tool_name == "spotify_search_tracks":
            data = spotify.search_tracks(
                query=arguments["query"],
                limit=_clamp(arguments.get("limit", 10), 1, 50),
            )
        elif tool_name == "spotify_search_artists":
            data = spotify.search_artists(
                query=arguments["query"],
                limit=_clamp(arguments.get("limit", 5), 1, 50),
            )
        elif tool_name == "spotify_search_albums":
            data = spotify.search_albums(
                query=arguments["query"],
                limit=_clamp(arguments.get("limit", 5), 1, 50),
            )
        elif tool_name == "spotify_search_playlists":
            data = spotify.search_playlists(
                query=arguments["query"],
                limit=_clamp(arguments.get("limit", 5), 1, 50),
            )
        elif tool_name == "spotify_artist_albums":
            data = spotify.get_artist_albums(
                artist_id=arguments["artist_id"],
                limit=_clamp(arguments.get("limit", 10), 1, 20),
            )
        elif tool_name == "spotify_album_tracks":
            data = spotify.get_album_tracks(
                album_id=arguments["album_id"],
                limit=_clamp(arguments.get("limit", 20), 1, 50),
            )
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                },
            }

        text = json.dumps(data, indent=2)
        meta = _build_meta(tool_name, arguments)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": text}],
                "_meta": meta,
            },
        }

    except Exception as e:
        meta = _build_meta(tool_name, arguments, error=str(e))
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "isError": True,
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "_meta": meta,
            },
        }


def main_stdio():
    """Run the MCP server over stdio."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = handle_request(request)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except json.JSONDecodeError:
            error_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            sys.stdout.write(json.dumps(error_resp) + "\n")
            sys.stdout.flush()


def main_http(host: str = "127.0.0.1", port: int = 0):
    """Run the MCP server over HTTP so traffic traverses the network stack.

    Uses ``ThreadingHTTPServer`` so each request is handled in its own
    thread.  This prevents slow Spotify API calls from blocking the
    server and causing client-side read timeouts.

    When *port* is 0 the OS picks a free port.  The chosen port is printed to
    stdout as ``PORT=<n>`` so the parent process can discover it.
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn

    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    class MCPHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                request = json.loads(body)
                response = handle_request(request)

                if response is None:
                    # Notification — send 204 No Content
                    self.send_response(204)
                    self.end_headers()
                    return

                payload = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except json.JSONDecodeError:
                error_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }
                payload = json.dumps(error_resp).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        def log_message(self, fmt, *args):
            pass  # suppress per-request logging

    server = ThreadingHTTPServer((host, port), MCPHandler)
    actual_port = server.server_address[1]
    # Signal the chosen port to the parent process
    sys.stdout.write(f"PORT={actual_port}\n")
    sys.stdout.flush()
    server.serve_forever()


def main():
    """Entry point — stdio by default, --http for network transport."""
    if "--http" in sys.argv:
        host = "127.0.0.1"
        port = 0
        for arg in sys.argv[1:]:
            if arg.startswith("--host="):
                host = arg.split("=", 1)[1]
            elif arg.startswith("--port="):
                port = int(arg.split("=", 1)[1])
        main_http(host, port)
    else:
        main_stdio()


if __name__ == "__main__":
    main()
