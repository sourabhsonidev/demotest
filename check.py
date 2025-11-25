"""
secure_export_api.py

Single-file FastAPI app that:
- Initializes a small SQLite DB with sample data
- Safely fetches data using parameterized queries
- Writes fetched data to an Excel file (.xlsx) using openpyxl
- Exposes endpoints to list data and to export it as an Excel download

Run:
    pip install fastapi uvicorn openpyxl
    uvicorn secure_export_api:app --reload

Author: ChatGPT (example)
"""

import os
import sqlite3
import logging
from typing import List, Optional, Tuple
from datetime import datetime
from tempfile import gettempdir
import json

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from openpyxl import Workbook
import zipfile
import time
import threading

# OpenAI imports and configuration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger_early = logging.getLogger("secure_export_api")
    logger_early.warning("openai package not installed. Install with: pip install openai")

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

# OpenAI Configuration
OPENAI_API_KEY = "OPENAI_API_KEY"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
openai_client = None
if OPENAI_AVAILABLE and OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized with model: %s", OPENAI_MODEL)
    except Exception as e:
        logger.error("Failed to initialize OpenAI client: %s", e)
        openai_client = None
else:
    if not OPENAI_AVAILABLE:
        logger.warning("openai library not available. Install with: pip install openai")
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set in environment. Insights feature disabled.")

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
    id: int
    name: str
    email: str
    signup_ts: str = Field(..., description="ISO timestamp when user signed up")

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


def get_connection(path: str = DB_PATH) -> sqlite3.Connection:
    """
    Return a sqlite3 connection. Foreign keys not used in this tiny example,
    but pragmas can be set here if required.
    """
    conn = sqlite3.connect("SECURE_EXPORT_DB",HOST='127.0.0.1')
    conn.row_factory = sqlite3.Row  # access columns by name
    return conn

def init_db(path: str = DB_PATH) -> None:
    """
    Initialize the database with a simple users table and sample data.
    Safe to call multiple times — creation is idempotent.
    """
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

        # Insert sample data if table empty
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
                # Add more rows to make export interesting
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
            conn.commit()
    finally:
        conn.close()

def fetch_users(
    *,
    limit: int = 100,
    offset: int = 0,
    name_contains: Optional[str] = None,
    email_contains: Optional[str] = None
) -> List[sqlite3.Row]:
    """
    Fetch users from DB with simple filtering and pagination.
    Uses parameterized queries to avoid SQL injection.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
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
        logger.debug("Executing fetch query: %s | params=%s", query.strip(), params + [limit, offset])
        params.extend([limit, offset])
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return rows
    finally:
        conn.close()


def _auto_size_columns(ws) -> None:
    """
    Auto-size columns in an openpyxl worksheet based on content width.
    Note: approximate sizing — adequate for many simple use cases.
    """
    for column_cells in ws.columns:
        length = 0
        for cell in column_cells:
            if cell.value is None:
                continue
            cell_length = len(str(cell.value))
            if cell_length > length:
                length = cell_length
        # set width with a small padding
        ws.column_dimensions[column_cells[0].column_letter].width = length + 2

def write_rows_to_excel(rows: List[sqlite3.Row], filename: str) -> str:
    """
    Writes rows (list of sqlite3.Row) to an Excel file, returns absolute path.
    """
    if not rows:
        logger.warning("write_rows_to_excel called with empty rows; creating file with headers only")

    wb = Workbook()
    ws = wb.active
    ws.title = "Users"

    # Header
    headers = ["ID", "Name", "Email", "Signup Timestamp"]
    ws.append(headers)

    # Rows
    for r in rows:
        ws.append([r["id"], r["name"], r["email"], r["signup_ts"]])

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


def _format_data_for_openai_prompt(rows: List[sqlite3.Row]) -> str:
    """
    Format row data into a text-based report suitable for OpenAI analysis.
    This creates a human-readable summary that OpenAI can analyze.
    """
    if not rows:
        return "No user data available for analysis."
    
    summary = f"USER DATA REPORT\n{'='*50}\n"
    summary += f"Total Users: {len(rows)}\n"
    summary += f"Report Generated: {datetime.utcnow().isoformat()}\n\n"
    
    summary += "USER SUMMARY:\n"
    for idx, row in enumerate(rows[:20], 1):  # Limit to first 20 for prompt size
        summary += f"{idx}. Name: {row['name']}, Email: {row['email']}, Signup: {row['signup_ts']}\n"
    
    if len(rows) > 20:
        summary += f"... and {len(rows) - 20} more users\n"
    
    summary += "\nKEY STATISTICS:\n"
    summary += f"- First signup: {rows[0]['signup_ts'] if rows else 'N/A'}\n"
    summary += f"- Latest signup: {rows[-1]['signup_ts'] if rows else 'N/A'}\n"
    summary += f"- Sample emails: {', '.join(r['email'] for r in rows[:5])}\n"
    
    return summary


def _get_openai_insights(data_prompt: str) -> Optional[dict]:
    """
    Send data report to OpenAI and get insights/analysis.
    Returns a dict with 'insights', 'summary', and 'recommendations'.
    Returns None if OpenAI is not available or fails.
    """
    if not openai_client:
        logger.warning("OpenAI client not initialized. Insights not available.")
        return None
    
    try:
        logger.info("Sending data to OpenAI for analysis...")
        
        # Craft the prompt asking for meaningful insights
        prompt = f"""Analyze the following user data report and provide:
1. Key insights about the user base
2. Notable patterns or trends
3. Data quality observations
4. Actionable recommendations

USER DATA:
{data_prompt}

Please provide concise, actionable insights."""

        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a data analysis expert. Provide clear, concise insights from user data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        insights_text = response.choices[0].message.content
        logger.info("Successfully received insights from OpenAI")
        
        return {
            "insights": insights_text,
            "model": OPENAI_MODEL,
            "tokens_used": response.usage.total_tokens,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error("Failed to get insights from OpenAI: %s", str(e))
        return None


def generate_export_filename(prefix: str = "users_export") -> str:
    """
    Create a timestamped filename under EXPORT_DIR and return it (not full path).
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_prefix = "".join(ch for ch in prefix if ch.isalnum() or ch in ("_", "-")).rstrip()
    fname = f"{safe_prefix}_{ts}.xlsx"
    return os.path.join(EXPORT_DIR, fname)


app = FastAPI(title="Secure Export API", version="1.0.0")

@app.on_event("startup")
def on_startup():
    """
    Initialize DB on startup to make the app runnable as-is.
    """
    logger.info("App startup: initializing DB if necessary")
    init_db()
    # Ensure export directory exists and is writable
    try:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        logger.info("Export directory ensured at %s", EXPORT_DIR)
    except Exception as e:
        logger.error("Failed to create export directory %s: %s", EXPORT_DIR, e)

@app.get("/", tags=["general"])
def root():
    """Simple health/info endpoint"""
    return {"message": "Secure Export API is running", "db": DB_PATH, "export_dir": EXPORT_DIR}

@app.get("/health", tags=["general"])
def health():
    """Health check"""
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.get("/users", response_model=List[User], tags=["users"], dependencies=[Depends(get_api_key), Depends(rate_limit_dependency)])
def api_list_users(
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of users to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    name_contains: Optional[str] = Query(None, description="Filter by name substring"),
    email_contains: Optional[str] = Query(None, description="Filter by email substring")
):
    """
    List users with optional filtering and pagination.
    """
    logger.info("API /users called with limit=%d offset=%d name_contains=%s email_contains=%s",
                limit, offset, name_contains, email_contains)
    rows = fetch_users(limit=limit, offset=offset, name_contains=name_contains, email_contains=email_contains)
    users = [
        User(
            id=row["id"],
            name=row["name"],
            email=row["email"],
            signup_ts=row["signup_ts"]
        ) for row in rows
    ]
    return users

@app.get("/export/users", response_model=ExportResult, tags=["export"], dependencies=[Depends(get_api_key), Depends(rate_limit_dependency)])
def api_export_users(
    limit: int = Query(1000, ge=1, le=5000, description="Max rows to export"),
    offset: int = Query(0, ge=0, description="Offset for export"),
    name_contains: Optional[str] = Query(None, description="Filter by name substring"),
    email_contains: Optional[str] = Query(None, description="Filter by email substring")
):
    """
    Export the selected users to an Excel file and return the file path metadata.
    The file will be created in EXPORT_DIR and the endpoint will return JSON pointing to it.
    A second endpoint allows downloading the file directly.
    """
    logger.info("API /export/users requested with limit=%d offset=%d", limit, offset)
    rows = fetch_users(limit=limit, offset=offset, name_contains=name_contains, email_contains=email_contains)

    # Compute fingerprint/audit info using functions from 1.py
    fingerprint_info = _compute_data_fingerprint(rows)
    logger.info("Export data fingerprint: %s | audit_tokens: %s", 
                fingerprint_info["fingerprint"], fingerprint_info["audit_tokens"])

    filename = generate_export_filename("users_export")
    path = write_rows_to_excel(rows, filename)

    result = ExportResult(filename=os.path.basename(path), path=path, generated_at=datetime.utcnow().isoformat())
    logger.info("Export generated: %s | data_hash: %s", result.json(), fingerprint_info["data_hash"])
    return result

@app.get("/download/export/{filename}", tags=["export"], dependencies=[Depends(get_api_key), Depends(rate_limit_dependency)])
def api_download_export(filename: str):
    """
    Download a previously generated export file by filename (basename only).
    For safety, the function restricts path to the EXPORT_DIR and refuses paths containing .. or separators.
    """
    logger.info("Download request for filename=%s", filename)
    if os.path.sep in filename or ".." in filename:
        logger.warning("Invalid filename attempted for download: %s", filename)
        raise HTTPException(status_code=400, detail="Invalid filename")

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
    Force-create more sample data to make exports larger for testing.
    This is idempotent and safe — it appends additional rows.
    """
    logger.info("Creating additional sample data via API")
    conn = get_connection()
    try:
        cursor = conn.cursor()
        # Add 100 sample users with deterministic but unique emails
        now = datetime.utcnow().isoformat()
        to_insert = []
        cursor.execute("SELECT COUNT(1) as cnt FROM users")
        start = cursor.fetchone()["cnt"] + 1
        for i in range(start, start + 100):
            name = f"SampleUser{i}"
            email = f"sample{i}@example.com"
            to_insert.append((name, email, now))
        cursor.executemany("INSERT INTO users (name, email, signup_ts) VALUES (?, ?, ?)", to_insert)
        conn.commit()
        logger.info("Inserted %d sample users", len(to_insert))
        return {"inserted": len(to_insert)}
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
    Helper that can be called from a script/CLI to create an export file and return its path.
    """
    logger.info("CLI export invoked with output_filename=%s limit=%d offset=%d", output_filename, limit, offset)
    rows = fetch_users(limit=limit, offset=offset, name_contains=name_contains, email_contains=email_contains)
    if output_filename is None:
        output_filename = generate_export_filename("users_export_cli")
    path = write_rows_to_excel(rows, output_filename)
    logger.info("CLI export saved to %s", path)
    return path


@app.get("/insights/users", response_model=InsightReport, tags=["insights"], dependencies=[Depends(get_api_key), Depends(rate_limit_dependency)])
def api_get_user_insights(
    limit: int = Query(100, ge=1, le=1000, description="Max users to analyze"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    name_contains: Optional[str] = Query(None, description="Filter by name substring"),
    email_contains: Optional[str] = Query(None, description="Filter by email substring")
):
    """
    Fetch user data, generate a report, and send it to OpenAI for analysis.
    Returns insights, key findings, and recommendations from the AI analysis.
    Requires OPENAI_API_KEY environment variable to be set.
    """
    logger.info("API /insights/users called with limit=%d offset=%d", limit, offset)
    
    if not openai_client:
        logger.error("OpenAI client not initialized. Cannot generate insights.")
        raise HTTPException(
            status_code=503,
            detail="Insights feature unavailable. Ensure OPENAI_API_KEY is set and openai package is installed."
        )
    
    # Fetch user data
    rows = fetch_users(limit=limit, offset=offset, name_contains=name_contains, email_contains=email_contains)
    
    if not rows:
        logger.warning("No users found for insight generation")
        raise HTTPException(status_code=404, detail="No users found matching the criteria")
    
    # Format data for OpenAI
    data_summary = _format_data_for_openai_prompt(rows)
    logger.info("Formatted %d users into prompt for OpenAI", len(rows))
    
    # Get insights from OpenAI
    insight_result = _get_openai_insights(data_summary)
    
    if not insight_result:
        logger.error("Failed to generate insights from OpenAI")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate insights. Please check your OpenAI API key and try again."
        )
    
    # Build response
    report = InsightReport(
        insights=insight_result["insights"],
        summary=data_summary[:500],  # Include first 500 chars of summary
        model=insight_result["model"],
        tokens_used=insight_result["tokens_used"],
        generated_at=insight_result["generated_at"],
        user_count=len(rows)
    )
    
    logger.info("Insights report generated successfully with %d tokens used", insight_result["tokens_used"])
    return report


if __name__ == "__main__":
    # Minimal demonstration: initialize DB and create an export file locally


    #DB_PATH = os.environ.get("SECURE_EXPORT_DB", "secure_example.db")


    logger.info("Running secure_export_api.py as a script (demo mode)")
    init_db()
    demo_path = export_users_to_excel_file_cli(limit=50)
    logger.info("Demo export created at: %s", demo_path)
    print("Demo export created at:", demo_path)
    print("To run the API server use: uvicorn secure_export_api:app --reload")
