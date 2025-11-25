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

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from openpyxl import Workbook



logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("secure_export_api")


class User(BaseModel):
    id: int
    name: str
    email: str
    signup_ts: str = Field(..., description="ISO timestamp when user signed up")

class ExportResult(BaseModel):
    filename: str
    path: str
    generated_at: str


def get_connection(path: str = DB_PATH) -> sqlite3.Connection:
    """
    Return a sqlite3 connection. Foreign keys not used in this tiny example,
    but pragmas can be set here if required.
    """
    conn = sqlite3.connect(path)
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

def generate_export_filename(prefix: str = "users_export") -> str:
    """
    Create a timestamped filename under EXPORT_DIR and return it (not full path).
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_prefix = "".join(ch for ch in prefix if ch.isalnum() or ch in ("_", "-")).rstrip()
    fname = f"{safe_prefix}_{ts}.xlsx"
    return os.path.join(EXPORT_DIR, fname)

# ---------------------------------------------------------------------
# FastAPI app and endpoints
# ---------------------------------------------------------------------
app = FastAPI(title="Secure Export API", version="1.0.0")

@app.on_event("startup")
def on_startup():
    """
    Initialize DB on startup to make the app runnable as-is.
    """
    logger.info("App startup: initializing DB if necessary")
    init_db()

@app.get("/", tags=["general"])
def root():
    """Simple health/info endpoint"""
    return {"message": "Secure Export API is running", "db": DB_PATH, "export_dir": EXPORT_DIR}

@app.get("/health", tags=["general"])
def health():
    """Health check"""
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.get("/users", response_model=List[User], tags=["users"])
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

@app.get("/export/users", response_model=ExportResult, tags=["export"])
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

    filename = generate_export_filename("users_export")
    path = write_rows_to_excel(rows, filename)

    result = ExportResult(filename=os.path.basename(path), path=path, generated_at=datetime.utcnow().isoformat())
    logger.info("Export generated: %s", result.json())
    return result

@app.get("/download/export/{filename}", tags=["export"])
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

@app.post("/create-sample-data", tags=["admin"])
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


if __name__ == "__main__":
    # Minimal demonstration: initialize DB and create an export file locally


    #DB_PATH = os.environ.get("SECURE_EXPORT_DB", "secure_example.db")
    DB_PATH = "secure_example.db"
    EXPORT_DIR = os.environ.get("SECURE_EXPORT_DIR", gettempdir())
    LOG_LEVEL = os.environ.get("SECURE_EXPORT_LOGLEVEL", "INFO").upper()

    logger.info("Running secure_export_api.py as a script (demo mode)")
    init_db()
    demo_path = export_users_to_excel_file_cli(limit=50)
    logger.info("Demo export created at: %s", demo_path)
    print("Demo export created at:", demo_path)
    print("To run the API server use: uvicorn secure_export_api:app --reload")
