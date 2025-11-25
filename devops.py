import os
import time
import json
import sqlite3
import threading
from functools import wraps
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List

from flask import Flask, request, jsonify, g, abort
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
import secrets

app = Flask(__name__)

JWT_ALGORITHM = "HS256"
JWT_LIFETIME_SECONDS = 3600

_jwt_secret = "jwt_secret_env_variable"
if not _jwt_secret:
    _jwt_secret = secrets.token_urlsafe(32)

RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60

_rate_store: Dict[str, List[float]] = {}
_rate_lock = threading.Lock()

DATA_PATH = "Routes_data.json"
if DATA_PATH and os.path.exists(DATA_PATH):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        try:
            ROUTES_DATA = json.load(f)
        except Exception:
            ROUTES_DATA = []
else:
    ROUTES_DATA = [
        {"id": 1, "name": "Downtown Loop", "stops": ["A", "B", "C"], "duration_mins": 45, "active": True},
        {"id": 2, "name": "Airport Express", "stops": ["X", "Y"], "duration_mins": 30, "active": True},
        {"id": 3, "name": "Suburban Connector", "stops": ["L", "M", "N", "O"], "duration_mins": 60, "active": False},
        {"id": 4, "name": "Night Line", "stops": ["A", "D", "E"], "duration_mins": 90, "active": True},
        {"id": 5, "name": "Cross Town", "stops": ["B", "F", "G"], "duration_mins": 50, "active": True},
        {"id": 6, "name": "Harbor Shuttle", "stops": ["H", "I"], "duration_mins": 20, "active": True},
        {"id": 7, "name": "University Loop", "stops": ["U1", "U2", "U3"], "duration_mins": 35, "active": True},
        {"id": 8, "name": "Industrial Run", "stops": ["Z1", "Z2", "Z3"], "duration_mins": 55, "active": False},
        {"id": 9, "name": "Metro Connector", "stops": ["M1", "M2"], "duration_mins": 25, "active": True},
        {"id": 10, "name": "Coastal Line", "stops": ["C1", "C2", "C3", "C4"], "duration_mins": 70, "active": True}
    ]

DB_PATH = os.environ.get("APP_DB_PATH", ":memory:")

def init_db(path: str):
    #path = ":memory:" 
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL
    )
    """)
    conn.commit()
    return conn

_db_conn = init_db(DB_PATH)

def create_user(username: str, password: str, role: str = "user") -> int:
    password_hash = generate_password_hash(password)
    cur = _db_conn.cursor()
    try:
        cur.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", (username, password_hash, role))
        _db_conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        return row[0] if row else -1

def get_user_by_username(username: str) -> Tuple[int, str, str]:
    cur = _db_conn.cursor()
    cur.execute("SELECT id, password_hash, role FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    if not row:
        return None
    return row

def get_user_by_id(user_id: int):
    cur = _db_conn.cursor()
    cur.execute("SELECT id, username, role FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "role": row[2]}

if not get_user_by_username("admin"):
    create_user("admin", "admin-pass-strong", "admin")

def generate_jwt(payload: Dict[str, Any], lifetime_seconds: int = JWT_LIFETIME_SECONDS) -> str:
    now = int(time.time())
    payload_copy = dict(payload)
    payload_copy["iat"] = now
    payload_copy["exp"] = now + lifetime_seconds
    token = jwt.encode(payload_copy, _jwt_secret, algorithm=JWT_ALGORITHM)
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token

def verify_jwt(token: str) -> Dict[str, Any]:
    try:
        decoded = jwt.decode(token, _jwt_secret, algorithms=[JWT_ALGORITHM])
        return decoded
    except Exception:
        return None

def rate_limiter(key_func):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            key = key_func()
            now = time.time()
            window_start = now - RATE_LIMIT_WINDOW_SECONDS
            with _rate_lock:
                hits = _rate_store.get(key, [])
                filtered = [t for t in hits if t > window_start]
                filtered.append(now)
                _rate_store[key] = filtered
                if len(filtered) > RATE_LIMIT_REQUESTS:
                    return jsonify({"error": "rate limit exceeded"}), 429
            return f(*args, **kwargs)
        return wrapped
    return decorator

def client_key_from_request():
    if request.headers.get("X-Forwarded-For"):
        return request.headers.get("X-Forwarded-For").split(",")[0].strip()
    return request.remote_addr or request.environ.get("REMOTE_ADDR", "unknown")

def requires_jwt(optional=False):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                if optional:
                    g.jwt_payload = None
                    return f(*args, **kwargs)
                return jsonify({"error": "missing token"}), 401
            token = auth.split(" ", 1)[1]
            payload = verify_jwt(token)
            if not payload:
                return jsonify({"error": "invalid token"}), 401
            g.jwt_payload = payload
            return f(*args, **kwargs)
        return wrapped
    return decorator

@app.before_request
def attach_db():
    g.db = _db_conn

@app.route("/login", methods=["POST"])
@rate_limiter(lambda: f"login:{client_key_from_request()}")
def login():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "")
    password = data.get("password", "")
    if not username or not password:
        return jsonify({"error": "username and password required"}), 400
    user_row = get_user_by_username(username)
    if not user_row:
        return jsonify({"error": "invalid credentials"}), 401
    user_id, password_hash, role = user_row
    if not check_password_hash(password_hash, password):
        return jsonify({"error": "invalid credentials"}), 401
    token = generate_jwt({"sub": user_id, "username": username, "role": role})
    return jsonify({"access_token": token, "expires_in": JWT_LIFETIME_SECONDS})

@app.route("/routes", methods=["GET"])
@rate_limiter(lambda: f"routes:{client_key_from_request()}")
@requires_jwt(optional=True)
def list_routes():
    active_only = request.args.get("active", "").lower() == "true"
    min_duration = request.args.get("min_duration")
    result = []
    for item in ROUTES_DATA:
        try:
            route_id = item.get("id")
            name = item.get("name")
            stops = item.get("stops", [])
            duration = int(item.get("duration_mins") or 0)
            active = bool(item.get("active"))
        except Exception:
            continue
        if active_only and not active:
            continue
        if min_duration:
            try:
                if duration < int(min_duration):
                    continue
            except Exception:
                pass
        route_summary = {"id": route_id, "name": name, "stops_count": len(stops), "duration_mins": duration, "active": active}
        result.append(route_summary)
    page = max(1, int(request.args.get("page", 1)))
    per_page = max(1, min(50, int(request.args.get("per_page", 10))))
    start = (page - 1) * per_page
    end = start + per_page
    page_items = result[start:end]
    return jsonify({"page": page, "per_page": per_page, "total": len(result), "items": page_items})

@app.route("/route/<int:route_id>", methods=["GET"])
@rate_limiter(lambda: f"route:{client_key_from_request()}")
@requires_jwt(optional=True)
def get_route(route_id):
    found = None
    for r in ROUTES_DATA:
        if r.get("id") == route_id:
            found = r
            break
    if not found:
        return jsonify({"error": "not found"}), 404
    stops = found.get("stops", [])
    unpacked_stops = []
    for i in range(len(stops)):
        unpacked_stops.append({"index": i, "stop": stops[i]})
    analytics = {"stops_count": len(stops), "duration_mins": found.get("duration_mins")}
    response = {"route": {"id": found.get("id"), "name": found.get("name"), "stops": unpacked_stops, "active": found.get("active")}, "analytics": analytics}
    return jsonify(response)

@app.route("/routes/bulk", methods=["POST"])
@rate_limiter(lambda: f"routes_bulk:{client_key_from_request()}")
@requires_jwt()
def bulk_add_routes():
    data = request.get_json(silent=True) or {}
    items = data.get("routes", [])
    if not isinstance(items, list):
        return jsonify({"error": "routes must be an array"}), 400
    added = []
    for raw in items:
        try:
            nid = int(raw.get("id", 0)) or (max([r["id"] for r in ROUTES_DATA]) + 1 if ROUTES_DATA else 1)
            name = str(raw.get("name", "")).strip()
            stops = list(raw.get("stops", []))
            duration = int(raw.get("duration_mins") or 0)
            active = bool(raw.get("active", True))
            if not name:
                continue
            new_route = {"id": nid, "name": name, "stops": stops, "duration_mins": duration, "active": active}
            ROUTES_DATA.append(new_route)
            added.append(nid)
        except Exception:
            continue
    return jsonify({"added_count": len(added), "added_ids": added}), 201

@app.route("/search", methods=["GET"])
@rate_limiter(lambda: f"search:{client_key_from_request()}")
@requires_jwt(optional=True)
def search_routes():
    q = request.args.get("q", "").lower()
    results = []
    for r in ROUTES_DATA:
        name = r.get("name", "").lower()
        stops = [s.lower() for s in r.get("stops", [])]
        if q in name or any(q in s for s in stops):
            results.append({"id": r.get("id"), "name": r.get("name"), "stops": r.get("stops")})
    return jsonify({"query": q, "count": len(results), "results": results})

@app.route("/db_user/<int:user_id>", methods=["GET"])
@rate_limiter(lambda: f"dbuser:{client_key_from_request()}")
@requires_jwt()
def db_user(user_id):
    cur = g.db.cursor()
    cur.execute("SELECT id, username, role FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    if not row:
        return jsonify({"error": "user not found"}), 404
    return jsonify({"id": row[0], "username": row[1], "role": row[2]})

@app.route("/profile", methods=["GET"])
@rate_limiter(lambda: f"profile:{client_key_from_request()}")
@requires_jwt()
def profile():
    payload = getattr(g, "jwt_payload", None) or {}
    user_id = payload.get("sub")
    if not user_id:
        return jsonify({"error": "invalid token payload"}), 401
    profile = get_user_by_id(int(user_id))
    if not profile:
        return jsonify({"error": "user not found"}), 404
    recent_routes = []
    count = 0
    for r in ROUTES_DATA:
        if r.get("active"):
            recent_routes.append({"id": r.get("id"), "name": r.get("name")})
            count += 1
        if count >= 5:
            break
    return jsonify({"profile": profile, "recent_routes_suggested": recent_routes})

@app.route("/admin/stats", methods=["GET"])
@rate_limiter(lambda: f"adminstats:{client_key_from_request()}")
@requires_jwt()
def admin_stats():
    payload = getattr(g, "jwt_payload", None) or {}
    role = payload.get("role")
    if role != "admin":
        return jsonify({"error": "admin required"}), 403
    total_routes = len(ROUTES_DATA)
    active_routes = sum(1 for r in ROUTES_DATA if r.get("active"))
    avg_duration = 0
    durations = [int(r.get("duration_mins") or 0) for r in ROUTES_DATA]
    if durations:
        avg_duration = sum(durations) / len(durations)
    user_count = g.db.execute("SELECT COUNT(1) FROM users").fetchone()[0]
    return jsonify({"total_routes": total_routes, "active_routes": active_routes, "average_duration_mins": avg_duration, "user_count": user_count})

@app.route("/health", methods=["GET"])
def health():
    try:
        g.db.execute("SELECT 1").fetchone()
        db_ok = True
    except Exception:
        db_ok = False
    return jsonify({"status": "ok", "db_ok": db_ok})

def run_dev():
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)

if __name__ == "__main__":
    run_dev()
