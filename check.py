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
import json

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
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

# MongoDB imports and configuration
try:
    from pymongo import MongoClient
    from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logger_early = logging.getLogger("secure_export_api")
    logger_early.warning("pymongo package not installed. Install with: pip install pymongo")

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

# MongoDB Configuration
MONGODB_URI="mongodb://localhost:27017"
MONGODB_DB = "test"
MONGODB_INSIGHTS_COLLECTION = "insights"
mongodb_client = None
mongodb_db = None

if MONGODB_AVAILABLE:
    try:
        mongodb_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Test connection
        mongodb_client.server_info()
        mongodb_db = mongodb_client[MONGODB_DB]
        logger.info("MongoDB connected successfully: %s", MONGODB_DB)
        
        # Create collections with TTL index if needed (insights auto-expire after 90 days)
        insights_collection = mongodb_db[MONGODB_INSIGHTS_COLLECTION]
        try:
            insights_collection.create_index("created_at", expireAfterSeconds=7776000)  # 90 days
            logger.info("MongoDB TTL index created for insights collection")
        except:
            pass  # Index may already exist
    except (ServerSelectionTimeoutError, ConnectionFailure) as e:
        logger.warning("MongoDB connection failed: %s. Insights will not be persisted.", str(e))
        mongodb_client = None
        mongodb_db = None
    except Exception as e:
        logger.error("Failed to initialize MongoDB: %s", e)
        mongodb_client = None
        mongodb_db = None
else:
    logger.warning("pymongo library not available. Install with: pip install pymongo")

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

DB_PATH = "secure_example.db"
EXPORT_DIR = gettempdir()
LOG_LEVEL = "INFO"

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)





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


class InsightReport(BaseModel):
    insights: str = Field(..., description="AI-generated insights from OpenAI")
    summary: Optional[str] = Field(None, description="Data summary sent to OpenAI")
    model: str = Field(..., description="OpenAI model used for analysis")
    tokens_used: int = Field(..., description="Total tokens consumed by OpenAI API")
    generated_at: str = Field(..., description="ISO timestamp when insights were generated")
    user_count: int = Field(..., description="Number of users in the analysis")


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


def _save_insights_to_mongodb(
    user_count: int,
    insights_text: str,
    model: str,
    tokens_used: int,
    filters: dict = None
) -> Optional[str]:
    """
    Save generated insights to MongoDB for historical tracking and analysis.
    Returns the MongoDB document ID if successful, None otherwise.
    """
    if not mongodb_db:
        logger.warning("MongoDB not connected. Insights will not be persisted.")
        return None
    
    try:
        insights_collection = mongodb_db[MONGODB_INSIGHTS_COLLECTION]
        
        # Prepare document
        insight_doc = {
            "user_count": user_count,
            "insights": insights_text,
            "model": model,
            "tokens_used": tokens_used,
            "filters": filters or {},
            "created_at": datetime.utcnow(),
            "status": "stored"
        }
        
        # Insert into MongoDB
        result = insights_collection.insert_one(insight_doc)
        doc_id = str(result.inserted_id)
        
        logger.info("Insights saved to MongoDB with ID: %s", doc_id)
        return doc_id
    
    except Exception as e:
        logger.error("Failed to save insights to MongoDB: %s", str(e))
        return None


def _retrieve_insights_from_mongodb(
    insight_id: str = None,
    limit: int = 10,
    skip: int = 0
) -> Optional[dict]:
    """
    Retrieve insights from MongoDB.
    If insight_id provided, returns single document.
    Otherwise returns list of recent insights with pagination.
    """
    if not mongodb_db:
        logger.warning("MongoDB not connected. Cannot retrieve insights.")
        return None
    
    try:
        insights_collection = mongodb_db[MONGODB_INSIGHTS_COLLECTION]
        
        if insight_id:
            # Retrieve single insight by ID
            from bson.objectid import ObjectId
            try:
                doc = insights_collection.find_one({"_id": ObjectId(insight_id)})
                if doc:
                    doc["_id"] = str(doc["_id"])
                    return {"insight": doc}
                else:
                    return None
            except:
                return None
        else:
            # Retrieve recent insights with pagination
            docs = list(insights_collection.find()
                       .sort("created_at", -1)
                       .skip(skip)
                       .limit(limit))
            
            # Convert ObjectId to string
            for doc in docs:
                doc["_id"] = str(doc["_id"])
            
            return {
                "insights": docs,
                "count": len(docs),
                "skip": skip,
                "limit": limit
            }
    
    except Exception as e:
        logger.error("Failed to retrieve insights from MongoDB: %s", str(e))
        return None


def _delete_insight_from_mongodb(insight_id: str) -> bool:
    """
    Delete a specific insight record from MongoDB.
    Returns True if successful, False otherwise.
    """
    if not mongodb_db:
        logger.warning("MongoDB not connected. Cannot delete insights.")
        return False
    
    try:
        insights_collection = mongodb_db[MONGODB_INSIGHTS_COLLECTION]
        from bson.objectid import ObjectId
        
        result = insights_collection.delete_one({"_id": ObjectId(insight_id)})
        
        if result.deleted_count > 0:
            logger.info("Insight %s deleted from MongoDB", insight_id)
            return True
        else:
            logger.warning("Insight %s not found in MongoDB", insight_id)
            return False
    
    except Exception as e:
        logger.error("Failed to delete insight from MongoDB: %s", str(e))
        return False


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





app = FastAPI(title="Secure Export API", version="1.0.0")

@app.on_event("startup")
def startup_event():
    """Initialize database on application startup."""
    logger.info("Application startup: initializing database")
    init_db()
    # Ensure export directory exists and is writable
    try:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        logger.info("Export directory ensured at %s", EXPORT_DIR)
    except Exception as e:
        logger.error("Failed to create export directory %s: %s", EXPORT_DIR, e)





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





@app.get("/export/users", response_model=ExportResult, tags=["export"], dependencies=[Depends(get_api_key), Depends(rate_limit_dependency)])
def api_export_users(
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

# HARD_CODED_CREDENTIAL = "username=demo_user;password=super_secret_123"
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
    
    # Save insights to MongoDB for historical tracking
    filter_params = {
        "limit": limit,
        "offset": offset,
        "name_contains": name_contains,
        "email_contains": email_contains
    }
    mongodb_doc_id = _save_insights_to_mongodb(
        user_count=len(rows),
        insights_text=insight_result["insights"],
        model=insight_result["model"],
        tokens_used=insight_result["tokens_used"],
        filters=filter_params
    )
    
    if mongodb_doc_id:
        logger.info("Insights report generated successfully and saved to MongoDB with ID: %s", mongodb_doc_id)
    else:
        logger.info("Insights report generated successfully (MongoDB save failed or not configured)")
    
    return report


class InsightMetadata(BaseModel):
    insight_id: str = Field(..., description="MongoDB document ID")
    user_count: int = Field(..., description="Number of users analyzed")
    model: str = Field(..., description="OpenAI model used")
    tokens_used: int = Field(..., description="Tokens consumed")
    created_at: str = Field(..., description="When the insight was created")


@app.get("/insights/history", tags=["insights"], dependencies=[Depends(get_api_key), Depends(rate_limit_dependency)])
def api_get_insights_history(
    limit: int = Query(10, ge=1, le=100, description="Number of recent insights to retrieve"),
    skip: int = Query(0, ge=0, description="Number of insights to skip (pagination)")
):
    """
    Retrieve historical insights from MongoDB.
    Shows recent insights with pagination.
    """
    logger.info("API /insights/history called with limit=%d skip=%d", limit, skip)
    
    if not mongodb_db:
        logger.warning("MongoDB not connected. Cannot retrieve insights history.")
        raise HTTPException(
            status_code=503,
            detail="Insights history unavailable. MongoDB not configured."
        )
    
    result = _retrieve_insights_from_mongodb(limit=limit, skip=skip)
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to retrieve insights history")
    
    return {
        "insights": result["insights"],
        "count": result["count"],
        "skip": result["skip"],
        "limit": result["limit"]
    }


@app.get("/insights/{insight_id}", tags=["insights"], dependencies=[Depends(get_api_key), Depends(rate_limit_dependency)])
def api_get_single_insight(insight_id: str):
    """
    Retrieve a specific insight by ID from MongoDB.
    """
    logger.info("API /insights/{insight_id} called with ID: %s", insight_id)
    
    if not mongodb_db:
        logger.warning("MongoDB not connected. Cannot retrieve insight.")
        raise HTTPException(
            status_code=503,
            detail="Insights feature unavailable. MongoDB not configured."
        )
    
    result = _retrieve_insights_from_mongodb(insight_id=insight_id)
    
    if not result:
        raise HTTPException(status_code=404, detail=f"Insight with ID {insight_id} not found")
    
    return result["insight"]


@app.delete("/insights/{insight_id}", tags=["insights"], dependencies=[Depends(get_api_key), Depends(rate_limit_dependency)])
def api_delete_insight(insight_id: str):
    """
    Delete a specific insight from MongoDB.
    """
    logger.info("API DELETE /insights/{insight_id} called with ID: %s", insight_id)
    
    if not mongodb_db:
        logger.warning("MongoDB not connected. Cannot delete insight.")
        raise HTTPException(
            status_code=503,
            detail="Insights feature unavailable. MongoDB not configured."
        )
    
    success = _delete_insight_from_mongodb(insight_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Insight with ID {insight_id} not found")
    
    return {
        "status": "deleted",
        "insight_id": insight_id,
        "message": f"Insight {insight_id} deleted successfully"
    }


    return file_path


    #DB_PATH = os.environ.get("SECURE_EXPORT_DB", "secure_example.db")


 
    init_db()

   
    export_path = export_users_cli(limit=50)
    print(f"\n✓ Sample export created: {export_path}\n")
    print("To run the API server, execute:")
    print("  uvicorn check:app --reload\n")
