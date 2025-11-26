"""
secure_export_api_flask.py

Flask-based rewrite of the Secure Export API. Provides:
- SQLite initialization and safe queries
- Excel export (openpyxl) and ZIP creation
- JWT-based authentication (/auth/login)
- Rate limiting via Flask-Limiter (60 requests/min per user/IP)
- CORS enabled for local development origins

Run:
    pip install flask flask-jwt-extended flask-limiter flask-cors openpyxl
    python secure_export_api_flask.py

Author: Conversion from FastAPI by automated assistant
"""

import os
import sqlite3
import logging
from datetime import datetime
from tempfile import gettempdir
from typing import List, Optional
import zipfile

from flask import Flask, request, jsonify, send_file, abort
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from openpyxl import Workbook


# Basic configuration and logging
DB_PATH = os.environ.get("SECURE_EXPORT_DB", "secure_example.db")
EXPORT_DIR = os.environ.get("SECURE_EXPORT_DIR", os.path.join(gettempdir(), "secure_exports"))
LOG_LEVEL = os.environ.get("SECURE_EXPORT_LOGLEVEL", "INFO")
API_KEY = os.environ.get("SECURE_EXPORT_API_KEY", "SECURE_EXPORT_API_KEY")
JWT_SECRET = os.environ.get("SECURE_EXPORT_JWT_SECRET", "please-change-me")

numeric_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(level=numeric_level, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("secure_export_api_flask")


def get_connection(path: str = DB_PATH) -> sqlite3.Connection:
    """Return a sqlite3 connection. Use the provided path."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: str = DB_PATH) -> None:
    logger.info("Initializing database at %s", path)
    conn = get_connection(path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                signup_ts TEXT NOT NULL
            )
        """)
        conn.commit()

        cursor.execute("SELECT COUNT(1) as cnt FROM users")
        count = cursor.fetchone()["cnt"]
        if count == 0:
            logger.info("Inserting sample users into the database")
            sample_users = [
                ("Alice Smith", "alice@example.com", datetime.utcnow().isoformat()),
                ("Bob Johnson", "bob@example.com", datetime.utcnow().isoformat()),
                ("Carol Williams", "carol@example.com", datetime.utcnow().isoformat()),
                ("David Brown", "david@example.com", datetime.utcnow().isoformat()),
                ("Eve Davis", "eve@example.com", datetime.utcnow().isoformat()),
            ]
            cursor.executemany(
                "INSERT INTO users (name, email, signup_ts) VALUES (?, ?, ?)",
                sample_users
            )
            conn.commit()
    finally:
        conn.close()


def fetch_users(limit: int = 100, offset: int = 0, name_contains: Optional[str] = None, email_contains: Optional[str] = None) -> List[sqlite3.Row]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        where_clauses = []
        params: List = []
        if name_contains:
            where_clauses.append("name LIKE ?")
            params.append(f"%{name_contains}%")
        if email_contains:
            where_clauses.append("email LIKE ?")
            params.append(f"%{email_contains}%")
        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        query = f"""
            SELECT id, name, email, signup_ts
            FROM users
            {where_sql}
            ORDER BY id ASC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        logger.debug("Executing fetch query: %s | params=%s", query.strip(), params)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return rows
    finally:
        conn.close()


def _auto_size_columns(ws) -> None:
    for column_cells in ws.columns:
        length = 0
        for cell in column_cells:
            if cell.value is None:
                continue
            cell_length = len(str(cell.value))
            if cell_length > length:
                length = cell_length
        ws.column_dimensions[column_cells[0].column_letter].width = length + 2


def write_rows_to_excel(rows: List[sqlite3.Row], filename: str) -> str:
    wb = Workbook()
    ws = wb.active
    ws.title = "Users"
    headers = ["ID", "Name", "Email", "Signup Timestamp"]
    ws.append(headers)
    for r in rows:
        ws.append([r["id"], r["name"], r["email"], r["signup_ts"]])
    _auto_size_columns(ws)
    abs_path = os.path.abspath(filename)
    wb.save(abs_path)
    logger.info("Saved Excel file to %s", abs_path)
    return abs_path


def generate_export_filename(prefix: str = "users_export") -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_prefix = "".join(ch for ch in prefix if ch.isalnum() or ch in ("_", "-")).rstrip()
    fname = f"{safe_prefix}_{ts}.xlsx"
    return os.path.join(EXPORT_DIR, fname)


def zip_export_file(excel_path: str) -> str:
    base_name = os.path.basename(excel_path)
    zip_filename = base_name.replace('.xlsx', '.zip')
    zip_path = os.path.join(EXPORT_DIR, zip_filename)
    logger.info("Creating ZIP archive: %s", zip_path)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(excel_path, arcname=base_name)
    return zip_path


# Flask app setup
app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = JWT_SECRET
jwt = JWTManager(app)

# CORS configuration for local dev
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:5000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:5000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8080",
    "http://0.0.0.0:3000",
    "http://0.0.0.0:8080",
]
CORS(app, origins=CORS_ALLOWED_ORIGINS, supports_credentials=True)


def limiter_key_func():
    # Prefer JWT identity if present, otherwise fall back to remote IP
    try:
        ident = get_jwt_identity()
        if ident:
            return f"user:{ident}"
    except Exception:
        pass
    return get_remote_address()


limiter = Limiter(app, key_func=limiter_key_func, default_limits=["60 per minute"])


@app.before_first_request
def on_startup():
    logger.info("Flask app startup: initializing DB and export dir")
    init_db()
    os.makedirs(EXPORT_DIR, exist_ok=True)
    logger.info("Export directory: %s", EXPORT_DIR)


@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Secure Export API (Flask) is running", "db": DB_PATH, "export_dir": EXPORT_DIR})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})


@app.route('/auth/login', methods=['POST'])
def login():
    """Exchanges an API key for a short-lived JWT access token.
    Request JSON: { "api_key": "..." }
    """
    data = request.get_json(silent=True) or {}
    api_key = data.get('api_key') or request.headers.get('X-API-KEY')
    if not api_key:
        return jsonify({"msg": "Missing API key"}), 400
    if api_key != API_KEY:
        logger.warning("Invalid API key attempted via /auth/login")
        return jsonify({"msg": "Invalid API key"}), 401

    # Create a token with identity==api_key (or could be a username)
    access_token = create_access_token(identity=api_key)
    return jsonify(access_token=access_token)


@app.route('/users', methods=['GET'])
@jwt_required(optional=True)
@limiter.limit("60 per minute")
def api_list_users():
    # Protected: jwt_required(optional=True) allows rate-limiting by user if present
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))
    name_contains = request.args.get('name_contains')
    email_contains = request.args.get('email_contains')
    logger.info("API /users called limit=%s offset=%s name_contains=%s email_contains=%s", limit, offset, name_contains, email_contains)
    rows = fetch_users(limit=limit, offset=offset, name_contains=name_contains, email_contains=email_contains)
    users = [dict(id=r['id'], name=r['name'], email=r['email'], signup_ts=r['signup_ts']) for r in rows]
    return jsonify(users)


@app.route('/export/users', methods=['GET'])
@jwt_required()
@limiter.limit("60 per minute")
def api_export_users():
    limit = int(request.args.get('limit', 1000))
    offset = int(request.args.get('offset', 0))
    name_contains = request.args.get('name_contains')
    email_contains = request.args.get('email_contains')
    logger.info("API /export/users requested limit=%s offset=%s", limit, offset)
    rows = fetch_users(limit=limit, offset=offset, name_contains=name_contains, email_contains=email_contains)
    filename = generate_export_filename('users_export')
    path = write_rows_to_excel(rows, filename)
    result = {"filename": os.path.basename(path), "path": path, "generated_at": datetime.utcnow().isoformat()}
    logger.info("Export generated: %s", result)
    return jsonify(result)


@app.route('/download/export/<path:filename>', methods=['GET'])
@jwt_required()
@limiter.limit("60 per minute")
def api_download_export(filename: str):
    logger.info("Download request for filename=%s", filename)
    if os.path.sep in filename or '..' in filename:
        logger.warning("Invalid filename attempted for download: %s", filename)
        return jsonify({"msg": "Invalid filename"}), 400
    full_path = os.path.join(EXPORT_DIR, filename)
    if not os.path.exists(full_path):
        logger.error("Requested file does not exist: %s", full_path)
        return jsonify({"msg": "File not found"}), 404
    return send_file(full_path, as_attachment=True)


@app.route('/export/users/zip', methods=['GET'])
@jwt_required()
@limiter.limit("60 per minute")
def api_export_users_zip():
    limit = int(request.args.get('limit', 1000))
    offset = int(request.args.get('offset', 0))
    name_contains = request.args.get('name_contains')
    email_contains = request.args.get('email_contains')
    logger.info("API /export/users/zip called")
    rows = fetch_users(limit=limit, offset=offset, name_contains=name_contains, email_contains=email_contains)
    excel_filename = generate_export_filename('users_export')
    excel_path = write_rows_to_excel(rows, excel_filename)
    zip_path = zip_export_file(excel_path)
    response = {"excel_file": os.path.basename(excel_path), "zip_file": os.path.basename(zip_path), "zip_path": zip_path, "generated_at": datetime.utcnow().isoformat()}
    logger.info("ZIP export complete: %s", response)
    return jsonify(response)





if __name__ == '__main__':
    # Ensure export dir exists
    os.makedirs(EXPORT_DIR, exist_ok=True)
    logger.info("Starting Flask Secure Export API on http://127.0.0.1:8000")
    # Default host/port chosen to be 0.0.0.0:8000 for local dev
    app.run(host='0.0.0.0', port=8000, debug=(LOG_LEVEL == 'DEBUG'))
