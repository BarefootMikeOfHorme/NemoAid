import asyncio
import json
import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_agent():
    # Setup server parameters to run our local mcp_server.py script
    server_params = StdioServerParameters(
        command="C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
        args=["mcp_server.py"],
        env=None
    )

    print("Connecting to local MCP SQL server...", flush=True)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 1. Discover available tools from our MCP server
            tools_response = await session.list_tools()
            ollama_tools = []
            
            for tool in tools_response.tools:
                ollama_tools.append({
                    'type': 'function',
                    'function': {
                        'name': tool.name,
                        'description': tool.description,
                        'parameters': tool.inputSchema
                    }
                })

            print(f"Discovered tools: {[t.name for t in tools_response.tools]}", flush=True)

            # User prompt requiring SQL database access
            prompt = "Who works in the Engineering department and what are their salaries?"
            print(f"\nUser Prompt: {prompt}\n", flush=True)

            messages = [{'role': 'user', 'content': prompt}]

            # 2. Send prompt + tools to Ollama (make sure you have a tool-capable model pulled like qwen2.5 or llama3.1)
            response = ollama.chat(
                model='qwen2.5', 
                messages=messages,
                tools=ollama_tools
            )

            # 3. Handle model tool-call requests
            if response.get('message', {}).get('tool_calls'):
                for tool_call in response['message']['tool_calls']:
                    tool_name = tool_call['function']['name']
                    tool_args = tool_call['function']['arguments']
                    
                    print(f"🤖 Model decided to call tool: {tool_name} with args: {tool_args}", flush=True)

                    # Execute the tool via MCP session
                    result = await session.call_tool(tool_name, tool_args)
                    tool_output = result.content[0].text
                    print(f"🛠️ Tool Execution Output: {tool_output}", flush=True)

                    # Append history and query model back for final answer
                    messages.append(response['message'])
                    messages.append({
                        'role': 'tool',
                        'content': tool_output,
                    })

                    final_response = ollama.chat(
                        model='qwen2.5',
                        messages=messages
                    )
                    
                    print("\n--- FINAL AGENT ANSWER ---")
                    print(final_response['message']['content'])
            else:
                print("Model response (No tool called):")
                print(response['message']['content'])

if __name__ == "__main__":
    asyncio.run(run_agent())
