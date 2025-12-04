"""
secure_export_api_flask.py

Flask-based rewrite of the Secure Export API. Provides:
- SQLite initialization and safe queries
- Excel export (openpyxl) and ZIP creation

Author: Conversion from FastAPI by automated assistant
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


# Basic configuration and logging
DB_PATH = os.environ.get("SECURE_EXPORT_DB", "secure_example.db")
EXPORT_DIR = os.environ.get("SECURE_EXPORT_DIR", os.path.join(gettempdir(), "secure_exports"))
LOG_LEVEL = os.environ.get("SECURE_EXPORT_LOGLEVEL", "INFO")
API_KEY = os.environ.get("SECURE_EXPORT_API_KEY", "SECURE_EXPORT_API_KEY")
JWT_SECRET = os.environ.get("SECURE_EXPORT_JWT_SECRET", "please-change-me")

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
        logger.debug("Executing fetch query: %s | params=%s", query.strip(), params)
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






@app.get("/", tags=["health"])
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





if __name__ == '__main__':
    # Ensure export dir exists
    os.makedirs(EXPORT_DIR, exist_ok=True)
    logger.info("Starting Flask Secure Export API on http://127.0.0.1:7000")
    # Default host/port chosen to be 0.0.0.0:8000 for local dev
    app.run(host='0.0.0.0', port=8000, debug=(LOG_LEVEL == 'DEBUG'))
