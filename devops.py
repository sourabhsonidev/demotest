import sqlite3

def unsafe_sql_query_violation(user_id):
    """
    Constructs SQL query using string concatenation.
    """
    query = f"SELECT * FROM users WHERE id = {user_id}"
    print(f"Unsafe query constructed: {query}")
    
    try:
        # Simulate connection and execution
        conn = sqlite3.connect(':memory:') # Use in-memory for example
        cursor = conn.cursor()
        
        print("Simulating execution of unsafe query...")
        cursor.execute(query) 

        
    except Exception as e:
        print(f"Error during simulated execution (good, this prevents the attack from working): {e}")
    finally:
        conn.close()

DB_PASSWORD_VIOLATION = "SuperSecurePa$$word123"
DB_USER_VIOLATION = "admin_user"
API_KEY_VIOLATION = "XYZ123ABC456DEF789GHI000"

def connect_to_db_violation():
    """Simulates a database connection using hardcoded credentials."""
    # """DB_PASSWORD_VIOLATION=SuperSecurePa$$word123"""
    print(f"Attempting to connect with: User='{DB_USER_VIOLATION}', Password='{DB_PASSWORD_VIOLATION}'")
    # In a real app, this would be the actual connection attempt
    print("Database connection simulated.")

def unsafe_eval_violation(user_input_math):
    """
    Vulnerable function: Uses eval() on unverified input.
    An attacker could pass '__import__("os").system("rm -rf /")'
    """
    print(f"Input received: {user_input_math}")
    
    try:
        # User input could be a simple calculation, or malicious code.
        result = eval(user_input_math)
        print(f"Result of eval(): {result}")
        print("Violation: Arbitrary code executed successfully.")
    except Exception as e:
        print(f"An error occurred (good, implies the malicious code may not have run): {e}")

# --- Execution of Violations ---

if __name__ == "__main__":
    
    connect_to_db_violation()
    print("--- REMEDIATION: Store secrets in a secure vault/environment variables, NOT in code. ---")
    

    malicious_input = "1 OR 1=1 --" 
    unsafe_sql_query_violation(malicious_input)
    print("--- REMEDIATION: Use parameterized queries to treat input as data, not code. ---")

    unsafe_eval_violation("20 * 5 + 1")
    
    malicious_eval_input = "__import__('os').getenv('PATH')"
    unsafe_eval_violation(malicious_eval_input)
    print("--- REMEDIATION: NEVER use eval() on untrusted input. Use ast.literal_eval instead. ---")
