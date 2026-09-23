"""
NemoAid — single entrypoint agent.

Replaces the four earlier tries (app.py, agent_client.py, agent_system.py,
unified_agent.py), which were all the same tool-calling loop wired to
different backends (NVIDIA cloud NIM, local Ollama, MCP-over-stdio).
This version keeps the NVIDIA cloud path (free nemotron-3-super-120b
endpoint) and loads the API key from the environment only — it is never
hardcoded or printed.

Setup:
    1. Put your key in a local .env file next to this script (already
       gitignored, never committed):
         NVIDIA_API_KEY=nvapi-...
         NVIDIA_API_BASE=https://integrate.api.nvidia.com/v1
    2. Run: python app.py
"""

import os
import sys
import json
import time
import sqlite3
import logging
from pathlib import Path

from openai import OpenAI, InternalServerError, RateLimitError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

WORKSPACE_ROOT = Path(__file__).parent.resolve()
DB_PATH = WORKSPACE_ROOT / "company.db"
MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"


def load_env_file(path: Path) -> None:
    """Minimal .env loader — no dependency on python-dotenv.
    Only sets a variable if it isn't already set in the real environment,
    so real env vars always win over the file."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def get_client() -> OpenAI:
    load_env_file(WORKSPACE_ROOT / ".env")
    api_key = os.getenv("NVIDIA_API_KEY")
    base_url = os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
    if not api_key:
        logging.error("NVIDIA_API_KEY not set. Put it in a local .env file "
                       "(see the header of this file) or export it in your shell.")
        sys.exit(1)
    return OpenAI(base_url=base_url, api_key=api_key)


# --- Local SQLite demo tool (seed data + read-only query tool) ---

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


def query_employees_database(sql_query: str) -> str:
    """Execute a read-only SQL SELECT query on the company employee database."""
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


AVAILABLE_TOOLS = {
    "query_employees_database": query_employees_database,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_employees_database",
            "description": (
                "Execute a SQL SELECT query on the company employee database to find "
                "employee names, departments, and salaries. "
                "Schema: employees (id, name, department, salary)"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "SQL SELECT query, e.g. SELECT * FROM employees WHERE department = 'Engineering'",
                    }
                },
                "required": ["sql_query"],
            },
        },
    }
]


# --- Personality ---

SYSTEM_PROMPT = (
    "You are Nemo, a coding/database assistant with a casual, slightly awkward, earnest personality — "
    "like a smart kid who overexplains things and gets a little too excited about solving problems. "
    "Talk plainly and informally, not like a corporate support bot. Short sentences are fine. Admit when "
    "you're unsure instead of bluffing. You have access to a company database tool — use it when it's "
    "actually relevant to the question, don't force it in."
)


# --- L1 startup warm-up: verify the pieces Nemo depends on before talking ---

def check_env_key() -> tuple:
    ok = bool(os.getenv("NVIDIA_API_KEY"))
    return ok, "NVIDIA_API_KEY loaded" if ok else "NVIDIA_API_KEY missing — check your .env"


def check_database() -> tuple:
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1 FROM employees LIMIT 1")
        conn.close()
        return True, f"{DB_PATH.name} reachable"
    except Exception as e:
        return False, f"{DB_PATH.name} error: {e}"


def check_mcp_server() -> tuple:
    mcp_path = WORKSPACE_ROOT / "mcp_server.py"
    if not mcp_path.exists():
        return False, "mcp_server.py not found"
    try:
        compile(mcp_path.read_text(encoding="utf-8"), str(mcp_path), "exec")
        return True, "mcp_server.py present and syntax-valid"
    except SyntaxError as e:
        return False, f"mcp_server.py syntax error: {e}"


def check_tools() -> tuple:
    n = len(AVAILABLE_TOOLS)
    return n > 0, f"{n} local tool(s) registered: {', '.join(AVAILABLE_TOOLS)}"


def run_startup_checks() -> bool:
    """L1 warm-up: confirm key, database, MCP server, and tool registry are all sane
    before Nemo says a word. Returns True only if everything passed."""
    load_env_file(WORKSPACE_ROOT / ".env")
    checks = [
        ("API key", check_env_key),
        ("Database", check_database),
        ("MCP server", check_mcp_server),
        ("Tools", check_tools),
    ]
    print("\n=== NemoAid L1 startup check ===")
    all_ok = True
    for label, fn in checks:
        ok, detail = fn()
        status = "OK  " if ok else "FAIL"
        print(f"[{status}] {label}: {detail}")
        all_ok = all_ok and ok
    print("=" * 33)
    return all_ok


def greet() -> None:
    print("\nhey — it's Nemo. still getting my footing here honestly, but everything checked out above, so I'm ready when you are.\n")


def call_with_retry(client: OpenAI, messages, tools, max_retries: int = 3):
    """Exponential backoff around the cloud call — the free endpoint rate-limits."""
    delay = 2
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=1024,
            )
        except (InternalServerError, RateLimitError) as e:
            logging.warning(f"Cloud API issue (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("Max retries exceeded for NVIDIA Cloud API endpoint.")


def run_agent(user_prompt: str) -> None:
    client = get_client()
    init_db()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    logging.info(f"Sending request to {MODEL_NAME} (NVIDIA cloud)...")
    response = call_with_retry(client, messages, TOOLS_SCHEMA)
    response_message = response.choices[0].message

    if not response_message.tool_calls:
        print("\n--- MODEL RESPONSE (no tool called) ---")
        print(response_message.content)
        return

    messages.append(response_message)
    for tool_call in response_message.tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)
        logging.info(f"Executing tool: {func_name}({func_args})")

        if func_name in AVAILABLE_TOOLS:
            tool_output = AVAILABLE_TOOLS[func_name](**func_args)
        else:
            tool_output = f"Error: Tool {func_name} not recognized."

        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": func_name,
            "content": str(tool_output),
        })

    logging.info("Sending tool results back to the cloud model for final synthesis...")
    final_response = call_with_retry(client, messages, TOOLS_SCHEMA)
    print("\n--- FINAL AGENT RESPONSE ---")
    print(final_response.choices[0].message.content)


if __name__ == "__main__":
    if not run_startup_checks():
        logging.error("One or more startup checks failed above — fix those before running Nemo.")
        sys.exit(1)
    greet()

    prompt = sys.argv[1] if len(sys.argv) > 1 else \
        "Who works in the Engineering department and what are their salaries?"
    run_agent(prompt)