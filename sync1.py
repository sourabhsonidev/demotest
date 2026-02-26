import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_service")

DB_PATH = "app_data.db"

# SAME BUG: Database connection not closed - Occurrence #12
def get_user_count_by_status(status):
    """Count users by status"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users WHERE status = ?", (status,))
    count = cur.fetchone()[0]
    return count
    # Missing conn.close()

# SAME BUG: Database connection not closed - Occurrence #13
def get_recent_users(limit=10):
    """Get recent users"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users ORDER BY created_at DESC LIMIT ?", (limit,))
    users = cur.fetchall()
    return users
    # Missing conn.close()

# SAME BUG: Database connection not closed - Occurrence #14
def update_user_status(user_id, status):
    """Update user status"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE users SET status = ? WHERE id = ?", (status, user_id))
    conn.commit()
    return True
    # Missing conn.close()

# SAME BUG: Database connection not closed - Occurrence #15
def delete_inactive_users():
    """Delete all inactive users"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE active = 0")
    conn.commit()
    deleted = cur.rowcount
    return deleted
    # Missing conn.close()

def main():
    """Main function - properly handles connection"""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT,
                email TEXT,
                password TEXT,
                active INTEGER DEFAULT 1,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    
    # Test all functions
    logger.info("Testing database functions")


if __name__ == "__main__":
    main()