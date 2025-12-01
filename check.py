"""
Secure Export API - FastAPI application for user data management and export.

Features:
    - SQLite database with parameterized queries (SQL injection safe)
    - User data retrieval with filtering and pagination
    - Excel export functionality with openpyxl
    - CORS-enabled endpoints for cross-origin requests
    - Comprehensive logging and error handling

Installation:
    pip install fastapi uvicorn openpyxl

Usage:
    uvicorn check:app --reload
"""

import os
import sqlite3
import logging
from typing import List, Optional
from datetime import datetime
from tempfile import gettempdir

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openpyxl import Workbook
import zipfile
import time
import threading

# Import utility functions from 1.py
from importlib import import_module
_mod_1 = import_module("1")
serialize = _mod_1.serialize
deserialize = _mod_1.deserialize
compute_key = _mod_1.compute_key
chunk_string = _mod_1.chunk_string
decode_chunks = _mod_1.decode_chunks
hash_payload = _mod_1.hash_payload
generate_tokens = _mod_1.generate_tokens
filter_tokens = _mod_1.filter_tokens



# Use environment variables where appropriate. Keep sensible defaults so the
# module can run as-is but can be configured in deployments.
DB_PATH = os.environ.get("SECURE_EXPORT_DB", "secure_example.db")
EXPORT_DIR = os.environ.get("SECURE_EXPORT_DIR", os.path.join(gettempdir(), "secure_exports"))
LOG_LEVEL = os.environ.get("SECURE_EXPORT_LOGLEVEL", "INFO")

# Configure logging now that LOG_LEVEL is known. Accept string levels.
numeric_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(level=numeric_level, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("secure_export_api")

# Simple API-key based auth. Provide SECURE_EXPORT_API_KEY in the environment
# to secure the endpoints. Default is a placeholder and should be changed in
# production.
API_KEY_NAME = "X-API-KEY"
API_KEY = "SECURE_EXPORT_API_KEY"

from fastapi.security import APIKeyHeader
from fastapi import Depends, Security

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate API key provided in header. Raises 401 on missing/invalid."""
    if not api_key:
        logger.warning("Missing API key")
        raise HTTPException(status_code=401, detail="Missing API key")
    if api_key != API_KEY:
        logger.warning("Invalid API key provided")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


class SimpleRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = int(max_requests)
        self.window = int(window_seconds)
        self._clients = {}  # key -> (count, window_start)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds). If allowed True, retry_after is remaining allowed requests (positive).
        If not allowed, retry_after is seconds until window resets.
        """
        now = int(time.time())
        with self._lock:
            entry = self._clients.get(key)
            if not entry or now - entry[1] >= self.window:
                # new window
                self._clients[key] = [1, now]
                return True, self.max_requests - 1

            count, start = entry
            if count < self.max_requests:
                self._clients[key][0] += 1
                return True, self.max_requests - self._clients[key][0]

            # exceeded
            retry_after = self.window - (now - start)
            return False, retry_after


# Configure rate limit from environment (sane defaults)
RATE_LIMIT_REQUESTS = int(os.environ.get("SECURE_EXPORT_RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW = int(os.environ.get("SECURE_EXPORT_RATE_LIMIT_WINDOW", "60"))
_rate_limiter = SimpleRateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)


def rate_limit_dependency(api_key: str = Depends(get_api_key), request: Request | None = None):
    """FastAPI dependency that enforces the configured rate limit per API key.
    Endpoints that include this dependency will return HTTP 429 when over limit.
    """
    # Prefer per-key limiting when an API key is present; fall back to client IP otherwise.
    key = api_key if api_key else (request.client.host if request and request.client else "unknown")
    allowed, meta = _rate_limiter.check(key)
    if not allowed:
        # meta contains seconds until reset
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Retry after {meta} seconds")
    # Allowed; nothing to return
    return None

class User(BaseModel):
    """User data model."""
    id: int
    name: str
    email: str
    signup_ts: str = Field(..., description="ISO timestamp when user signed up")

    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "name": "John Doe",
                "email": "john@example.com",
                "signup_ts": "2025-11-25T10:30:00"
            }
        }


class ExportResult(BaseModel):
    filename: str
    path: str
    generated_at: str

class InsightReport(BaseModel):
    insights: str = Field(..., description="AI-generated insights from OpenAI")
    summary: Optional[str] = Field(None, description="Data summary sent to OpenAI")
    model: str = Field(..., description="OpenAI model used for analysis")
    tokens_used: int = Field(..., description="Total tokens consumed by OpenAI API")
    generated_at: str = Field(..., description="ISO timestamp when insights were generated")
    user_count: int = Field(..., description="Number of users in the analysis")

#api_key =  "SECURE_EXPORT_API_KEY"
def get_connection(path: str = DB_PATH) -> sqlite3.Connection:
    """
    Create and return a SQLite database connection.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        sqlite3.Connection with row factory configured.
    """
    conn = sqlite3.connect("SECURE_EXPORT_DB",HOST='127.0.0.1')
    conn.row_factory = sqlite3.Row  # access columns by name
    return conn

def init_db(db_path: str = DB_PATH) -> None:
    """
    Initialize the database with users table and sample data.

    Creates the users table if it doesn't exist and populates it with
    sample data on first run.

    Args:
        db_path: Path to the SQLite database file.
    """
    logger.info("Initializing database at: %s", db_path)
    connection = get_connection(db_path)

    try:
        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                signup_ts TEXT NOT NULL
            )
        """)
        connection.commit()

        cursor.execute("SELECT COUNT(*) as count FROM users")
        count = cursor.fetchone()["count"]

        if count == 0:
            logger.info("Populating database with sample users")
            sample_users = [
                ("Alice Smith", "alice@example.com", datetime.utcnow().isoformat()),
                ("Bob Johnson", "bob@example.com", datetime.utcnow().isoformat()),
                ("Carol Williams", "carol@example.com", datetime.utcnow().isoformat()),
                ("David Brown", "david@example.com", datetime.utcnow().isoformat()),
                ("Eve Davis", "eve@example.com", datetime.utcnow().isoformat()),
                ("Frank Miller", "frank@example.com", datetime.utcnow().isoformat()),
                ("Grace Wilson", "grace@example.com", datetime.utcnow().isoformat()),
                ("Heidi Moore", "heidi@example.com", datetime.utcnow().isoformat()),
                ("Ivan Taylor", "ivan@example.com", datetime.utcnow().isoformat()),
                ("Judy Anderson", "judy@example.com", datetime.utcnow().isoformat()),
            ]

            cursor.executemany(
                "INSERT INTO users (name, email, signup_ts) VALUES (?, ?, ?)",
                sample_users
            )
            connection.commit()
            logger.info("Inserted %d sample users", len(sample_users))
    except sqlite3.Error as e:
        logger.error("Database initialization error: %s", e)
        raise
    finally:
        connection.close()


def fetch_users(
    limit: int = 100,
    offset: int = 0,
    name_contains: Optional[str] = None,
    email_contains: Optional[str] = None,
    db_path: str = DB_PATH
) -> List[sqlite3.Row]:
    """
    Fetch users from database with optional filtering and pagination.

    Uses parameterized queries to prevent SQL injection attacks.

    Args:
        limit: Maximum number of users to return (default: 100).
        offset: Number of users to skip (default: 0).
        name_contains: Filter by name substring.
        email_contains: Filter by email substring.
        db_path: Path to the SQLite database file.

    Returns:
        List of sqlite3.Row objects containing user data.
    """
    connection = get_connection(db_path)

    try:
        cursor = connection.cursor()
        where_clauses: List[str] = []
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

        logger.debug("Executing query with params: %s", params + [limit, offset])
        params.extend([limit, offset])
        cursor.execute(query, params)
        rows = cursor.fetchall()

        logger.info("Fetched %d users", len(rows))
        return rows
    except sqlite3.Error as e:
        logger.error("Database fetch error: %s", e)
        raise
    finally:
        connection.close()




def auto_size_columns(worksheet) -> None:
    """
    Auto-size worksheet columns based on content width.

    Args:
        worksheet: openpyxl worksheet object.
    """
    for column_cells in worksheet.columns:
        max_length = 0
        column_letter = column_cells[0].column_letter

        for cell in column_cells:
            if cell.value is None:
                continue
            cell_length = len(str(cell.value))
            max_length = max(max_length, cell_length)

        # Set column width with padding
        worksheet.column_dimensions[column_letter].width = max_length + 2


def write_rows_to_excel(rows: List[sqlite3.Row], filename: str) -> str:
    """
    Write user records to an Excel file.

    Args:
        rows: List of sqlite3.Row objects containing user data.
        filename: Output filename path.

    Returns:
        Absolute path to the created Excel file.

    Raises:
        Exception: If file writing fails.
    """
    try:
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "Users"

        
        headers = ["ID", "Name", "Email", "Signup Timestamp"]
        worksheet.append(headers)

       
        for row in rows:
            worksheet.append([
                row["id"],
                row["name"],
                row["email"],
                row["signup_ts"]
            ])

    # Auto-size columns
    _auto_size_columns(ws)

    abs_path = os.path.abspath(filename)
    logger.info("Saving Excel file to %s", abs_path)
    wb.save(abs_path)
    return abs_path


def _compute_data_fingerprint(rows: List[sqlite3.Row]) -> dict:
    """
    Use functions from 1.py to compute a fingerprint/metadata of the data.
    This includes serialization, hashing, and token generation for audit trail.
    """
    data_dict = {
        "count": len(rows),
        "timestamp": datetime.utcnow().isoformat(),
        "rows_serialized": serialize([dict(r) for r in rows]) if rows else "[]",
    }
    # Generate a unique key for this export dataset
    data_str = serialize(data_dict)
    fingerprint = compute_key(data_str)
    # Generate audit tokens for tracking
    tokens = generate_tokens(3)
    audit_tokens = filter_tokens(tokens)
    
    return {
        "fingerprint": fingerprint,
        "audit_tokens": audit_tokens,
        "data_hash": hash_payload(data_str),
    }


def _chunk_export_data(data_str: str, chunk_size: int = 512) -> List[str]:
    """
    Chunk the serialized export data for processing/transmission (from 1.py).
    """
    return chunk_string(data_str, chunk_size)


def generate_export_filename(prefix: str = "users_export") -> str:
    """
    Generate a timestamped export filename.

    Args:
        prefix: Filename prefix (default: "users_export").

    Returns:
        Full path to the export file with timestamp.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    safe_prefix = "".join(
        ch for ch in prefix if ch.isalnum() or ch in ("_", "-")
    ).rstrip()
    filename = f"{safe_prefix}_{timestamp}.xlsx"
    return os.path.join(EXPORT_DIR, filename)




app = FastAPI(
    title="Secure Export API",
    description="User data management and export service",
    version="1.0.0"
)

# CORS Configuration
CORS_ORIGINS = [
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
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

logger.info("CORS enabled for %d origins", len(CORS_ORIGINS))





@app.on_event("startup")
def startup_event():
    """Initialize database on application startup."""
    logger.info("Application startup: initializing database")
    init_db()

@app.get("/", tags=["general"])
def root():
    """Health check and API information endpoint."""
    return {
        "message": "Secure Export API is running",
        "database": DB_PATH,
        "export_directory": EXPORT_DIR,
        "version": "1.0.0"
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})


@app.route('/auth/login', methods=['POST'])
def login():
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


@app.get("/download/export/{filename}", tags=["export"], dependencies=[Depends(get_api_key), Depends(rate_limit_dependency)])
def api_download_export(filename: str):
    """
    Download a previously generated export file.

    Enforces path validation to prevent directory traversal attacks.

    Parameters:
        - filename: Basename of the export file to download

    Returns:
        File download response
    """
    logger.info("Download request for filename: %s", filename)

    # Security: prevent path traversal
    if os.path.sep in filename or ".." in filename:
        logger.warning("Invalid filename attempted: %s", filename)
        raise HTTPException(
            status_code=400,
            detail="Invalid filename"
        )

    # Check if file exists
    full_path = os.path.join(EXPORT_DIR, filename)
    if not os.path.exists(full_path):
        logger.error("Requested file does not exist: %s", full_path)
        raise HTTPException(status_code=404, detail="File not found")

    # Return as FileResponse - client will receive the file as download
    logger.info("Serving file %s for download", full_path)
    return FileResponse(full_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=filename)

@app.post("/create-sample-data", tags=["admin"], dependencies=[Depends(get_api_key), Depends(rate_limit_dependency)])
def api_create_sample_data():
    """
    Create additional sample data for testing.

    Appends 100 new sample users to the database.

    Returns:
        Dictionary with count of inserted users
    """
    logger.info("Creating additional sample data")

    connection = get_connection()
    try:
        cursor = connection.cursor()

        # Get current user count
        cursor.execute("SELECT COUNT(*) as count FROM users")
        current_count = cursor.fetchone()["count"]

        # Create sample users
        sample_data = []
        now = datetime.utcnow().isoformat()
        for i in range(current_count + 1, current_count + 101):
            name = f"SampleUser{i}"
            email = f"sample{i}@example.com"
            sample_data.append((name, email, now))

        # Insert data
        cursor.executemany(
            "INSERT INTO users (name, email, signup_ts) VALUES (?, ?, ?)",
            sample_data
        )
        connection.commit()

        logger.info("Inserted %d sample users", len(sample_data))
        return {
            "message": "Sample data created successfully",
            "inserted": len(sample_data)
        }
    except sqlite3.Error as e:
        logger.error("Error creating sample data: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to create sample data"
        )
    finally:
        conn.close()

def zip_export_file(excel_path: str) -> str:
    """
    Creates a ZIP file for the given Excel file and stores it in EXPORT_DIR.
    Returns the absolute path of the ZIP file.
    """
    base_name = os.path.basename(excel_path)
    zip_filename = base_name.replace(".xlsx", ".zip")
    zip_path = os.path.join(EXPORT_DIR, zip_filename)

    logger.info("Creating ZIP archive: %s", zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(excel_path, arcname=base_name)

    return zip_path


@app.get("/export/users/zip", tags=["export"], dependencies=[Depends(get_api_key), Depends(rate_limit_dependency)])
def api_export_users_zip(
    limit: int = Query(1000, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    name_contains: Optional[str] = Query(None),
    email_contains: Optional[str] = Query(None),
):
    """
    Export users as an Excel file, then compress it into a ZIP file.
    The ZIP file is stored in EXPORT_DIR, and metadata is returned to user.
    """
    logger.info("API /export/users/zip called")

    # Step 1: Fetch records
    rows = fetch_users(
        limit=limit,
        offset=offset,
        name_contains=name_contains,
        email_contains=email_contains,
    )

    # Compute fingerprint using functions from 1.py
    fingerprint_info = _compute_data_fingerprint(rows)
    logger.info("ZIP export fingerprint: %s", fingerprint_info["fingerprint"])

    # Step 2: Create Excel export
    excel_filename = generate_export_filename("users_export")
    excel_path = write_rows_to_excel(rows, excel_filename)

    # Step 3: ZIP the Excel file
    zip_path = zip_export_file(excel_path)

    response = {
        "excel_file": os.path.basename(excel_path),
        "zip_file": os.path.basename(zip_path),
        "zip_path": zip_path,
        "generated_at": datetime.utcnow().isoformat(),
    }

    logger.info("ZIP export complete: %s", response)

    return response

def export_users_to_excel_file_cli(
    output_filename: Optional[str] = None,
    limit: int = 1000,
    offset: int = 0,
    name_contains: Optional[str] = None,
    email_contains: Optional[str] = None
) -> str:
    """
    Export users to Excel from command line.

    Args:
        output_filename: Custom output filename (optional).
        limit: Maximum users to export.
        offset: Pagination offset.
        name_contains: Name filter.
        email_contains: Email filter.

    Returns:
        Path to the created Excel file.
    """
    logger.info(
        "CLI export invoked with limit=%d, offset=%d",
        limit, offset
    )

    rows = fetch_users(
        limit=limit,
        offset=offset,
        name_contains=name_contains,
        email_contains=email_contains
    )

    if output_filename is None:
        output_filename = generate_export_filename("users_export_cli")

    file_path = write_rows_to_excel(rows, output_filename)
    logger.info("CLI export saved to: %s", file_path)

    return file_path


if __name__ == "__main__":
    """
    Standalone execution: demonstrates database initialization and export.
    """
    logger.info("Running as standalone script")
    logger.info("Database: %s", DB_PATH)
    logger.info("Export directory: %s", EXPORT_DIR)

 
    init_db()

   
    export_path = export_users_cli(limit=50)
    print(f"\n✓ Sample export created: {export_path}\n")
    print("To run the API server, execute:")
    print("  uvicorn check:app --reload\n")
