"""
DevOps MongoDB helpers

Provides simple, well-documented MongoDB CRUD helpers that return
the affected or fetched documents. Designed for small DevOps scripts
and automation tasks.

Requirements:
    pip install pymongo

Usage:
    from devops import insert_one_document, find_documents
    doc = insert_one_document("my_collection", {"name": "x"})
    docs = find_documents("my_collection", {})
"""

from typing import Any, Dict, List, Optional
import logging
import os

from pymongo import MongoClient, ReturnDocument
from pymongo.results import InsertOneResult, InsertManyResult, UpdateResult, DeleteResult
from bson import ObjectId


# Configuration
MONGO_URI: str = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DEFAULT_DB: str = os.environ.get("MONGO_DEFAULT_DB", "test")

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


def insert_one_document(collection_name: str, document: Dict[str, Any], db_name: Optional[str] = None, client: Optional[MongoClient] = None) -> Dict[str, Any]:
    """Insert a single document and return it (with _id serialized).

    If client is not provided this function will create and close a client.
    """
    local_client = None
    try:
        if client is None:
            local_client = get_mongo_client()
            client = local_client

        coll = get_collection(collection_name, db_name=db_name, client=client)
        logger.debug("Inserting document into %s.%s: %s", db_name or DEFAULT_DB, collection_name, document)
        result: InsertOneResult = coll.insert_one(document)
        inserted = coll.find_one({"_id": result.inserted_id})
        return to_jsonable(inserted)
    except Exception:
        logger.exception("insert_one_document failed")
        raise
    finally:
        if local_client:
            local_client.close()


def insert_many_documents(collection_name: str, documents: List[Dict[str, Any]], db_name: Optional[str] = None, client: Optional[MongoClient] = None) -> List[Dict[str, Any]]:
    """Insert multiple documents and return the inserted documents."""
    local_client = None
    try:
        if client is None:
            local_client = get_mongo_client()
            client = local_client

        coll = get_collection(collection_name, db_name=db_name, client=client)
        logger.debug("Inserting %d documents into %s.%s", len(documents), db_name or DEFAULT_DB, collection_name)
        result: InsertManyResult = coll.insert_many(documents)
        inserted = list(coll.find({"_id": {"$in": result.inserted_ids}}))
        return [to_jsonable(d) for d in inserted]
    except Exception:
        logger.exception("insert_many_documents failed")
        raise
    finally:
        if local_client:
            local_client.close()


def find_one_document(collection_name: str, filter_query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, int]] = None, db_name: Optional[str] = None, client: Optional[MongoClient] = None) -> Optional[Dict[str, Any]]:
    """Return a single matching document (or None)."""
    local_client = None
    try:
        if client is None:
            local_client = get_mongo_client()
            client = local_client
        coll = get_collection(collection_name, db_name=db_name, client=client)
        doc = coll.find_one(filter_query or {}, projection)
        return to_jsonable(doc) if doc else None
    except Exception:
        logger.exception("find_one_document failed")
        raise
    finally:
        if local_client:
            local_client.close()


def find_documents(collection_name: str, filter_query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, int]] = None, limit: Optional[int] = None, skip: int = 0, db_name: Optional[str] = None, client: Optional[MongoClient] = None) -> List[Dict[str, Any]]:
    """Return a list of documents matching filter, with optional pagination."""
    local_client = None
    try:
        if client is None:
            local_client = get_mongo_client()
            client = local_client
        coll = get_collection(collection_name, db_name=db_name, client=client)
        cursor = coll.find(filter_query or {}, projection).skip(skip)
        if limit:
            cursor = cursor.limit(limit)
        docs = list(cursor)
        return [to_jsonable(d) for d in docs]
    except Exception:
        logger.exception("find_documents failed")
        raise
    finally:
        if local_client:
            local_client.close()


def update_one_document(collection_name: str, filter_query: Dict[str, Any], update_query: Dict[str, Any], upsert: bool = False, db_name: Optional[str] = None, client: Optional[MongoClient] = None) -> Optional[Dict[str, Any]]:
    """Update one document and return the updated document (or None)."""
    local_client = None
    try:
        if client is None:
            local_client = get_mongo_client()
            client = local_client
        coll = get_collection(collection_name, db_name=db_name, client=client)
        updated = coll.find_one_and_update(filter_query, update_query, upsert=upsert, return_document=ReturnDocument.AFTER)
        return to_jsonable(updated) if updated else None
    except Exception:
        logger.exception("update_one_document failed")
        raise
    finally:
        if local_client:
            local_client.close()


def update_many_documents(collection_name: str, filter_query: Dict[str, Any], update_query: Dict[str, Any], db_name: Optional[str] = None, client: Optional[MongoClient] = None) -> Dict[str, Any]:
    """Update many documents and return a summary and the updated docs."""
    local_client = None
    try:
        if client is None:
            local_client = get_mongo_client()
            client = local_client
        coll = get_collection(collection_name, db_name=db_name, client=client)
        result: UpdateResult = coll.update_many(filter_query, update_query)
        updated_docs = list(coll.find(filter_query))
        return {"matched_count": result.matched_count, "modified_count": result.modified_count, "updated_docs": [to_jsonable(d) for d in updated_docs]}
    except Exception:
        logger.exception("update_many_documents failed")
        raise
    finally:
        if local_client:
            local_client.close()


def delete_one_document(collection_name: str, filter_query: Dict[str, Any], db_name: Optional[str] = None, client: Optional[MongoClient] = None) -> Optional[Dict[str, Any]]:
    """Delete a single document and return the deleted document (or None)."""
    local_client = None
    try:
        if client is None:
            local_client = get_mongo_client()
            client = local_client
        coll = get_collection(collection_name, db_name=db_name, client=client)
        doc = coll.find_one(filter_query)
        if not doc:
            return None
        result: DeleteResult = coll.delete_one({"_id": doc["_id"]})
        return to_jsonable(doc)
    except Exception:
        logger.exception("delete_one_document failed")
        raise
    finally:
        if local_client:
            local_client.close()


def delete_many_documents(collection_name: str, filter_query: Dict[str, Any], db_name: Optional[str] = None, client: Optional[MongoClient] = None) -> Dict[str, Any]:
    """Delete many documents and return deleted count and list of deleted docs.

    WARNING: this loads all matched docs into memory before deleting.
    """
    local_client = None
    try:
        if client is None:
            local_client = get_mongo_client()
            client = local_client
        coll = get_collection(collection_name, db_name=db_name, client=client)
        docs = list(coll.find(filter_query))
        if not docs:
            return {"deleted_count": 0, "deleted_docs": []}
        result: DeleteResult = coll.delete_many(filter_query)
        return {"deleted_count": result.deleted_count, "deleted_docs": [to_jsonable(d) for d in docs]}
    except Exception:
        logger.exception("delete_many_documents failed")
        raise
    finally:
        if local_client:
            local_client.close()


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
