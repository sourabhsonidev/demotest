
from typing import Any, Dict, List, Optional
import logging
import os

from pymongo import MongoClient, ReturnDocument
from pymongo.results import InsertOneResult, InsertManyResult, UpdateResult, DeleteResult
from bson import ObjectId


# Configuration
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DEFAULT_DB = "test"

logging.basicConfig(level=os.environ.get("DEVOPS_LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def get_mongo_client(uri: Optional[str] = None) -> MongoClient:
    """Create and return a MongoClient for the given URI.

    The caller is responsible for closing the client if they create it.
    If functions in this module create a client internally they will close it.
    """
    _uri = uri or MONGO_URI
    logger.debug("Creating MongoClient for %s", _uri)
    return MongoClient(_uri)


def to_jsonable(obj: Any) -> Any:
    """Convert BSON types (ObjectId) to JSON-serializable values recursively."""
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def get_collection(collection_name: str, db_name: Optional[str] = None, client: Optional[MongoClient] = None):
    """Return the collection object for the given names.

    If `client` is None a temporary client will be created (and should be closed by caller).
    """
    _client = client or get_mongo_client()
    db = _client[db_name or DEFAULT_DB]
    return db[collection_name]



if __name__ == "__main__":
    # Example usage (requires a running MongoDB instance)
    coll = "devops_examples"

    try:
        inserted = insert_one_document(coll, {"name": "devops", "value": 1})
        print("Inserted:", inserted)

        found = find_one_document(coll, {"name": "devops"})
        print("Found:", found)

        updated = update_one_document(coll, {"name": "devops"}, {"$set": {"value": 2}})
        print("Updated:", updated)

        deleted = delete_one_document(coll, {"name": "devops"})
        print("Deleted:", deleted)
    except Exception as e:
        logger.exception("Example run failed: %s", e)
    print("--- REMEDIATION: NEVER use eval() on untrusted input. Use ast.literal_eval instead. ---")
