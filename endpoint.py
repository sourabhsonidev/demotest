import sqlite3
from fastapi import FastAPI, Query

app = FastAPI(title="Security Violations Demo API")


def unsafe_sql_query_violation(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    print(f"Unsafe query constructed: {query}")

    try:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE users (id TEXT, name TEXT)")
        cursor.execute("INSERT INTO users VALUES ('1', 'Alice')")
        cursor.execute("INSERT INTO users VALUES ('2', 'Bob')")
        conn.commit()

        print("Simulating execution of unsafe query...")
        cursor.execute(query)
        result = cursor.fetchall()
        return result

    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


DB_PASSWORD_VIOLATION = "SuperSecurePa$$word123"
DB_USER_VIOLATION = "admin_user"
API_KEY_VIOLATION = "XYZ123ABC456DEF789GHI000"


def connect_to_db_violation():
    return {
        "user": DB_USER_VIOLATION,
        "password": DB_PASSWORD_VIOLATION,
        "message": "This is UNSAFE — credentials should NOT be hardcoded!"
    }


def unsafe_eval_violation(user_input_math):
    print(f"Input received: {user_input_math}")

    try:
        result = eval(user_input_math)
        return {"result": result, "warning": "Eval executed — this is unsafe!"}
    except Exception as e:
        return {"error": str(e)}



def safe_sql_query(user_id):
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.execute("CREATE TABLE users (id TEXT, name TEXT)")
    cursor.execute("INSERT INTO users VALUES ('1', 'Alice')")
    cursor.execute("INSERT INTO users VALUES ('2', 'Bob')")
    conn.commit()

    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchall()

    conn.close()
    return result


import ast

def safe_eval(user_input_math):
    try:
        # literal_eval prevents code execution
        result = ast.literal_eval(user_input_math)
        return {"result": result}
    except Exception:
        return {"error": "Invalid input — only literals allowed."}



@app.get("/")
def home():
    return {"message": "Security Demo API Running"}

@app.get("/unsafe/db-credentials")
def api_unsafe_credentials():
    return connect_to_db_violation()


@app.get("/unsafe/sql")
def api_unsafe_sql(user_id: str = Query(...)):
    return unsafe_sql_query_violation(user_id)


@app.get("/unsafe/eval")
def api_unsafe_eval(expr: str = Query(...)):
    return unsafe_eval_violation(expr)

@app.get("/safe/sql")
def api_safe_sql(user_id: str = Query(...)):
    return safe_sql_query(user_id)


@app.get("/safe/eval")
def api_safe_eval(expr: str = Query(...)):
    return safe_eval(expr)


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
