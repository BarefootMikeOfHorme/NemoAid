import sqlite3
import json
from openai import OpenAI

# 1. Initialize NVIDIA Cloud Client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# 2. Setup a local SQLite Database & Seed Data
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

# 3. Define the actual Python function (Tool)
def query_employees_database(sql_query: str) -> str:
    """Execute a SQL SELECT query on the company employee database."""
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

# Map string names to actual callable Python functions
available_tools = {
    "query_employees_database": query_employees_database
}

# 4. Define OpenAI-compatible Tool Schema for NVIDIA NIM
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_employees_database",
            "description": "Execute a SQL SELECT query on the company employee database to find employee names, departments, and salaries. Schema: employees (id, name, department, salary)",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "The SQL SELECT query to execute, e.g., SELECT * FROM employees WHERE department = 'Engineering'"
                    }
                },
                "required": ["sql_query"]
            }
        }
    }
]

def run_agent_workflow():
    prompt = "Who works in the Engineering department and what are their salaries?"
    print(f"User Prompt: {prompt}\n", flush=True)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant with access to a company database tool."},
        {"role": "user", "content": prompt}
    ]

    print("Sending request to Nemotron-3 Super 120B (Cloud API)...", flush=True)
    
    # First turn: Ask the model (it will inspect tools and choose whether to call one)
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.2,
        max_tokens=1024
    )

    response_message = response.choices[0].message
    
    # Check if the model triggered a tool call
    if response_message.tool_calls:
        print("Model decided to call a tool.", flush=True)
        
        # Append assistant's intent to message history
        messages.append(response_message)

        for tool_call in response_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            
            print(f"-> Executing local function: {func_name} with arguments: {func_args}", flush=True)

            if func_name in available_tools:
                # Run local Python function
                tool_output = available_tools[func_name](**func_args)
                print(f"-> Database Result: {tool_output}", flush=True)

                # Feed the tool result back into the message array
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": func_name,
                    "content": str(tool_output)
                })

        print("\nSending tool results back to cloud model for final synthesis...", flush=True)
        
        # Second turn: Send tool results back so the model can write a natural language response
        final_response = client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b",
            messages=messages,
            temperature=0.2,
            max_tokens=1024
        )

        print("\n--- FINAL AGENT RESPONSE ---")
        print(final_response.choices[0].message.content)
    else:
        print("\n--- MODEL RESPONSE (No Tool Called) ---")
        print(response_message.content)

if __name__ == "__main__":
    run_agent_workflow()
