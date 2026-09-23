import os
import sys
import subprocess
import json
import logging
import time
from pathlib import Path
from openai import OpenAI, InternalServerError, RateLimitError

# Configure clean, readable output logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# 1. Secure Client Initialization (Loads strictly from NVIDIA standard env vars)
api_key = os.getenv("NVIDIA_API_KEY")
base_url = os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")

if not api_key:
    logging.error("SECURITY ERROR: The 'NVIDIA_API_KEY' environment variable is not set.")
    logging.error("Set it in PowerShell via: ='nvapi-your-key'")
    logging.error("Set it in WSL/Linux via: export NVIDIA_API_KEY='nvapi-your-key'")
    sys.exit(1)

client = OpenAI(base_url=base_url, api_key=api_key)
MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"

# Define Workspace Root safely for cross-platform (Windows & WSL Linux)
WORKSPACE_ROOT = Path(__file__).parent.resolve()

def validate_path(filepath: str) -> Path:
    """Ensures target path stays strictly inside workspace boundaries (Prevents Path Traversal)."""
    target_path = (WORKSPACE_ROOT / filepath).resolve()
    if not str(target_path).startswith(str(WORKSPACE_ROOT)):
        raise PermissionError("Security Violation: Access outside the workspace root is strictly prohibited.")
    return target_path

# --- SAFE TOOL IMPLEMENTATIONS ---

def secure_read_file(filepath: str) -> str:
    """Reads a file safely with path containment checks."""
    try:
        safe_path = validate_path(filepath)
        if not safe_path.exists():
            return f"Error: File '{filepath}' does not exist."
        if safe_path.name in [".env", "id_rsa", "credentials.json"] or ".git" in safe_path.parts:
            return "Security Error: Access to sensitive system or credential files is blocked."
        return safe_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Read Error: {str(e)}"

def secure_write_file(filepath: str, content: str) -> str:
    """Writes or updates a file safely within workspace boundaries."""
    try:
        safe_path = validate_path(filepath)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote and secured file: {filepath}"
    except Exception as e:
        return f"Write Error: {str(e)}"

def secure_run_command(command: str) -> str:
    """Executes permitted development commands with strict safety filtering."""
    forbidden_terms = ["rm -rf /", "del /f /s /q c:\\", "mkfs", "dd if=", ":(){ :|:& };:"]
    if any(term in command.lower() for term in forbidden_terms):
        return "Security Error: This command contains restricted destructive operations and was blocked."

    # Whitelist updated for Windows PowerShell & WSL Linux parity
    allowed_prefixes = ["git", "pytest", "python", "pip", "dir", "ls", "echo", "pwd", "get-location"]
    cmd_base = command.strip().split()[0].lower()
    
    if cmd_base not in allowed_prefixes:
        return f"Security Warning: Command '{cmd_base}' is not in the safe execution whitelist."

    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=30,
            cwd=str(WORKSPACE_ROOT)
        )
        output = result.stdout if result.returncode == 0 else result.stderr
        return output or "Command executed successfully with no output."
    except Exception as e:
        return f"Execution Error: {str(e)}"

available_tools = {
    "secure_read_file": secure_read_file,
    "secure_write_file": secure_write_file,
    "secure_run_command": secure_run_command
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "secure_read_file",
            "description": "Read code or text files from the local workspace directory safely.",
            "parameters": {
                "type": "object",
                "properties": {"filepath": {"type": "string", "description": "Relative path to the file inside the project workspace"}},
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "secure_write_file",
            "description": "Create or modify source code files within the project workspace sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Relative path to the file"},
                    "content": {"type": "string", "description": "Full file contents to write"}
                },
                "required": ["filepath", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "secure_run_command",
            "description": "Run whitelisted shell or git commands (e.g., git status, pwd, pytest, python script).",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The command string to execute"}},
                "required": ["command"]
            }
        }
    }
]

def call_nemotron_with_retry(messages, tools):
    """Wrapper with exponential backoff to handle transient cloud 500/rate limit errors gracefully."""
    max_retries = 3
    delay = 2
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2
            )
        except (InternalServerError, RateLimitError) as e:
            logging.warning(f"Cloud API Glitch (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2
    raise Exception("Max retries exceeded for NVIDIA Cloud API endpoint.")

def run_agent_workflow(user_prompt: str):
    messages = [
        {
            "role": "system",
            "content": (
                "You are Nemotron, an elite senior software engineering assistant. You operate under strict "
                "sandbox and security guardrails. Inspect repository status, evaluate code quality, and use "
                "your secure tools to assist the user cleanly without exposing credentials."
            )
        },
        {"role": "user", "content": user_prompt}
    ]

    print("\n" + "="*60)
    logging.info(f"AGENT REQUEST INITIATED: '{user_prompt}'")
    print("="*60 + "\n")

    for turn in range(1, 6):
        response = call_nemotron_with_retry(messages, tools_schema)
        response_message = response.choices[0].message
        messages.append(response_message)

        if not response_message.tool_calls:
            print("\n" + "="*60)
            print("🎯 FINAL AGENT RESPONSE:")
            print("="*60)
            print(response_message.content)
            print("="*60 + "\n")
            break

        for tool_call in response_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            
            logging.info(f"[Turn {turn}] Triggering Tool -> {func_name}({func_args})")

            if func_name in available_tools:
                tool_output = available_tools[func_name](**func_args)
            else:
                tool_output = f"Error: Tool {func_name} not recognized."
                
            # Print a neat summary of the tool output block
            preview = str(tool_output).strip().replace('\n', ' ')
            if len(preview) > 120:
                preview = preview[:120] + "..."
            logging.info(f"[Turn {turn}] Tool Output Result: {preview}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_output)
            })

if __name__ == "__main__":
    prompt = "Check the git status of our repository, read agent_system.py, and let me know if our credential security improvements are correct."
    run_agent_workflow(prompt)
