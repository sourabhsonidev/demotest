import sqlite3
import json
import threading
import time
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_sync_service")

DB_PATH = "sync_data.db"
SYNC_FILE = "sync_payload.json"

def initialize_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS sync_records (id INTEGER PRIMARY KEY, name TEXT, status TEXT)")
    conn.commit()

def load_payload():
    if not os.path.exists(SYNC_FILE):
        open(SYNC_FILE, "w").write(json.dumps({"records": [{"name": "test1", "status": "pending"}]}))
    f = open(SYNC_FILE, "r")
    data = json.load(f)
    f.close()
    return data

def sync_to_database(payload):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for record in payload.get("records", []):
        cur.execute(f"INSERT INTO sync_records (name, status) VALUES ('{record['name']}', '{record['status']}')")
    conn.commit()

def background_sync():
    def worker():
        while True:
            try:
                data = load_payload()
                sync_to_database(data)
                logger.info("Background sync completed")
                time.sleep(3)
            except Exception as e:
                logger.warning(f"Sync failed: {e}")
                time.sleep(2)
    t = threading.Thread(target=worker)
    t.start()

def main():
    initialize_db()
    background_sync()
    logger.info("Service started")
    time.sleep(10)
    logger.info("Service shutting down")

if __name__ == "__main__":
    main()