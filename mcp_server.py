"""
NemoAid MCP server.

Exposes tools over stdio (MCP protocol) so any MCP-aware client can use them:
  - query_employees_database — read-only SQL SELECT against company.db
  - secure_read_file / secure_write_file — sandboxed to the workspace root
  - secure_run_command — whitelisted shell/git commands only

The file/shell tools are folded in from secure_agent.py's guardrails
(path containment, credential-file blocklist, destructive-command blocklist,
command allowlist) so they behave identically whether called through
secure_agent.py directly or through this MCP server.
"""

import subprocess
import sqlite3
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("NemoAid-Server")

WORKSPACE_ROOT = Path(__file__).parent.resolve()
DB_PATH = WORKSPACE_ROOT / "company.db"


# --- SQL tool ---

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary INTEGER NOT NULL
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM employees")
    if cursor.fetchone()[0] == 0:
        cursor.executemany(
            "INSERT INTO employees (name, department, salary) VALUES (?, ?, ?)",
            [
                ("Alice Smith", "Engineering", 95000),
                ("Bob Jones", "Marketing", 72000),
                ("Charlie Brown", "Engineering", 105000),
                ("Diana Prince", "Management", 130000),
            ],
        )
        conn.commit()
    conn.close()


init_db()


@mcp.tool()
def query_employees_database(sql_query: str) -> str:
    """Execute a SQL SELECT query on the company employee database and return the results.
    Schema: employees (id, name, department, salary)
    Example: SELECT * FROM employees WHERE department = 'Engineering'
    """
    try:
        if not sql_query.strip().lower().startswith("select"):
            return "Error: Only SELECT queries are permitted for safety."
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        conn.close()
        return str(rows)
    except Exception as e:
        return f"SQL Error: {str(e)}"


# --- Sandboxed file/shell tools (folded in from secure_agent.py) ---

def validate_path(filepath: str) -> Path:
    """Ensures target path stays strictly inside the workspace root (prevents path traversal)."""
    target_path = (WORKSPACE_ROOT / filepath).resolve()
    if not str(target_path).startswith(str(WORKSPACE_ROOT)):
        raise PermissionError("Security Violation: Access outside the workspace root is strictly prohibited.")
    return target_path


@mcp.tool()
def secure_read_file(filepath: str) -> str:
    """Read a code or text file from the local workspace directory safely.
    filepath: relative path to the file inside the project workspace.
    """
    try:
        safe_path = validate_path(filepath)
        if not safe_path.exists():
            return f"Error: File '{filepath}' does not exist."
        if safe_path.name in [".env", "id_rsa", "credentials.json"] or ".git" in safe_path.parts:
            return "Security Error: Access to sensitive system or credential files is blocked."
        return safe_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Read Error: {str(e)}"


@mcp.tool()
def secure_write_file(filepath: str, content: str) -> str:
    """Create or modify a source code/text file within the project workspace sandbox.
    filepath: relative path to the file. content: full file contents to write.
    """
    try:
        safe_path = validate_path(filepath)
        if safe_path.name in [".env", "id_rsa", "credentials.json"] or ".git" in safe_path.parts:
            return "Security Error: Writing to sensitive system or credential files is blocked."
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote and secured file: {filepath}"
    except Exception as e:
        return f"Write Error: {str(e)}"


@mcp.tool()
def secure_run_command(command: str) -> str:
    """Run a whitelisted shell or git command (e.g., git status, pwd, pytest, python script).
    command: the command string to execute.
    """
    forbidden_terms = ["rm -rf /", "del /f /s /q c:\\", "mkfs", "dd if=", ":(){ :|:& };:"]
    if any(term in command.lower() for term in forbidden_terms):
        return "Security Error: This command contains restricted destructive operations and was blocked."

    allowed_prefixes = ["git", "pytest", "python", "pip", "dir", "ls", "echo", "pwd", "get-location"]
    cmd_base = command.strip().split()[0].lower() if command.strip() else ""

    if cmd_base not in allowed_prefixes:
        return f"Security Warning: Command '{cmd_base}' is not in the safe execution whitelist."

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(WORKSPACE_ROOT),
        )
        output = result.stdout if result.returncode == 0 else result.stderr
        return output or "Command executed successfully with no output."
    except Exception as e:
        return f"Execution Error: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")