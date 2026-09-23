import sqlite3
import ollama

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

def query_employees_database(sql_query: str) -> str:
    \"\"\"Execute a SQL SELECT query on the company employee database.\"\"\"
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

ollama_tools = [{
    'type': 'function',
    'function': {
        'name': 'query_employees_database',
        'description': 'Execute a SQL SELECT query on the company employee database to find employee details. Schema: employees (id, name, department, salary)',
        'parameters': {
            'type': 'object',
            'properties': {
                'sql_query': {
                    'type': 'string',
                    'description': 'The SQL SELECT query to execute, e.g. SELECT * FROM employees WHERE department = "Engineering"'
                }
            },
            'required': ['sql_query']
        }
    }
}]

def run_agent():
    prompt = "Who works in the Engineering department and what are their salaries?"
    print(f"User Prompt: {prompt}\n", flush=True)

    messages = [{'role': 'user', 'content': prompt}]

    print("Sending prompt and tools to local Ollama model (qwen2.5)...", flush=True)
    response = ollama.chat(
        model='qwen2.5',
        messages=messages,
        tools=ollama_tools
    )

    if response.get('message', {}).get('tool_calls'):
        for tool_call in response['message']['tool_calls']:
            func_name = tool_call['function']['name']
            func_args = tool_call['function']['arguments']
            
            print(f"🤖 Model called tool: {func_name} with args: {func_args}", flush=True)

            if func_name == "query_employees_database":
                tool_output = query_employees_database(**func_args)
                print(f"🛠️ Tool Execution Output: {tool_output}", flush=True)

                messages.append(response['message'])
                messages.append({
                    'role': 'tool',
                    'content': str(tool_output),
                })

                final_response = ollama.chat(
                    model='qwen2.5',
                    messages=messages
                )
                
                print("\n--- FINAL AGENT ANSWER ---")
                print(final_response['message']['content'])
    else:
        print("Model response:")
        print(response['message']['content'])

if __name__ == "__main__":
    run_agent()
