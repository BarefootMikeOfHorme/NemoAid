import sqlite3
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("SQL-Assistant-Server")

def init_db():
    conn = sqlite3.connect("company.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary INTEGER NOT NULL
        )
    ''')
    # Add dummy data if empty
    cursor.execute("SELECT COUNT(*) FROM employees")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("INSERT INTO employees (name, department, salary) VALUES (?, ?, ?)", [
            ("Alice Smith", "Engineering", 95000),
            ("Bob Jones", "Marketing", 72000),
            ("Charlie Brown", "Engineering", 105000),
            ("Diana Prince", "Management", 130000)
        ])
        conn.commit()
    conn.close()

init_db()

@mcp.tool()
def query_employees_database(sql_query: str) -> str:
    \"\"\"Execute a SQL SELECT query on the company employee database and return the results.
    Schema: employees (id, name, department, salary)
    Example: SELECT * FROM employees WHERE department = 'Engineering'
    \"\"\"
    try:
        if not sql_query.strip().lower().startswith("select"):
            return "Error: Only SELECT queries are permitted for safety."
            
        conn = sqlite3.connect("company.db")
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        conn.close()
        return str(rows)
    except Exception as e:
        return f"SQL Error: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
