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

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openpyxl import Workbook


DB_PATH: str = os.environ.get("SECURE_EXPORT_DB", "secure_example.db")
EXPORT_DIR: str = os.environ.get("SECURE_EXPORT_DIR", gettempdir())
LOG_LEVEL: str = os.environ.get("SECURE_EXPORT_LOGLEVEL", "INFO").upper()



logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)





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
    """Export operation result."""
    filename: str = Field(..., description="Exported file basename")
    path: str = Field(..., description="Absolute path to the exported file")
    generated_at: str = Field(..., description="ISO timestamp of generation")

    class Config:
        schema_extra = {
            "example": {
                "filename": "users_export_20251125T103000Z.xlsx",
                "path": "/tmp/users_export_20251125T103000Z.xlsx",
                "generated_at": "2025-11-25T10:30:00"
            }
        }




def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Create and return a SQLite database connection.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        sqlite3.Connection with row factory configured.
    """
    #db_path = "secure_example.db""
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


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

        auto_size_columns(worksheet)

        
        abs_path = os.path.abspath(filename)
        workbook.save(abs_path)
        logger.info("Excel file saved to: %s", abs_path)

        return abs_path
    except Exception as e:
        logger.error("Error writing Excel file: %s", e)
        raise


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





@app.get("/", tags=["health"])
def root():
    """Health check and API information endpoint."""
    return {
        "message": "Secure Export API is running",
        "database": DB_PATH,
        "export_directory": EXPORT_DIR,
        "version": "1.0.0"
    }


@app.get("/health", tags=["health"])
def health_check():
    """Health status endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }





@app.get("/users", response_model=List[User], tags=["users"])
def list_users(
    limit: int = Query(50, ge=1, le=1000, description="Max users to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    name_contains: Optional[str] = Query(
        None, description="Filter by name substring"
    ),
    email_contains: Optional[str] = Query(
        None, description="Filter by email substring"
    )
):
    """
    Retrieve a list of users with optional filtering and pagination.

    Parameters:
        - limit: Maximum number of users to return
        - offset: Number of users to skip
        - name_contains: Filter users by name
        - email_contains: Filter users by email

    Returns:
        List of User objects
    """
    logger.info(
        "GET /users called with limit=%d, offset=%d, "
        "name_contains=%s, email_contains=%s",
        limit, offset, name_contains, email_contains
    )

    try:
        rows = fetch_users(
            limit=limit,
            offset=offset,
            name_contains=name_contains,
            email_contains=email_contains
        )

        users = [
            User(
                id=row["id"],
                name=row["name"],
                email=row["email"],
                signup_ts=row["signup_ts"]
            )
            for row in rows
        ]
        return users
    except Exception as e:
        logger.error("Error listing users: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve users"
        )





@app.get("/export/users", response_model=ExportResult, tags=["export"])
def export_users(
    limit: int = Query(1000, ge=1, le=5000, description="Max rows to export"),
    offset: int = Query(0, ge=0, description="Export offset"),
    name_contains: Optional[str] = Query(
        None, description="Filter by name substring"
    ),
    email_contains: Optional[str] = Query(
        None, description="Filter by email substring"
    )
):
    """
    Export filtered users to an Excel file.

    Returns metadata about the generated export file.

    Parameters:
        - limit: Maximum number of users to export
        - offset: Pagination offset
        - name_contains: Filter by name
        - email_contains: Filter by email

    Returns:
        ExportResult containing file path and metadata
    """
    logger.info(
        "GET /export/users called with limit=%d, offset=%d",
        limit, offset
    )

    try:
        # Fetch users
        rows = fetch_users(
            limit=limit,
            offset=offset,
            name_contains=name_contains,
            email_contains=email_contains
        )

        # Generate export file
        filename = generate_export_filename("users_export")
        file_path = write_rows_to_excel(rows, filename)

        # Return result
        result = ExportResult(
            filename=os.path.basename(file_path),
            path=file_path,
            generated_at=datetime.utcnow().isoformat()
        )
        logger.info("Export created: %s", result.filename)
        return result
    except Exception as e:
        logger.error("Error exporting users: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to export users"
        )


@app.get("/download/export/{filename}", tags=["export"])
def download_export(filename: str):
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
        logger.error("File not found: %s", full_path)
        raise HTTPException(
            status_code=404,
            detail="Export file not found"
        )

    logger.info("Serving file for download: %s", full_path)
    return FileResponse(
        full_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename
    )





@app.post("/admin/sample-data", tags=["admin"])
def create_sample_data():
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
        connection.close()





def export_users_cli(
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
