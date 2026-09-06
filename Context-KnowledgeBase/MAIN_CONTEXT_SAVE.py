import ast
import json
import re
from pathlib import Path
from openai import OpenAI

# ---------------------------------------------------------
# 1. Local Tool Definitions
# ---------------------------------------------------------
BASE_DIR = Path("../my_knowledge_base")


def save_note_to_disk(
    content: str, filename: str, folder: str, tags: list[str] | str
) -> str:
  """Saves a specific learning concept or AI answer to a local markdown file."""
  if isinstance(tags, str):
    tags_str = tags.strip()
    if tags_str.startswith("[") and tags_str.endswith("]"):
      try:
        parsed = ast.literal_eval(tags_str)
        tags = parsed if isinstance(parsed, list) else [tags_str]
      except Exception:
        tags = [t.strip() for t in tags_str.strip("[]").split(",") if t.strip()]
    else:
      tags = [t.strip() for t in tags_str.split(",") if t.strip()]

  clean_folder = re.sub(r"[^\w\s-]", "", folder).strip()
  clean_filename = re.sub(r"[^\w\s.-]", "", filename).strip()
  if not clean_filename.endswith(".md"):
    clean_filename += ".md"

  target_dir = BASE_DIR / clean_folder
  target_dir.mkdir(parents=True, exist_ok=True)
  file_path = target_dir / clean_filename

  clean_tags = [re.sub(r"[^\w\s-]", "", str(tag)).strip() for tag in tags]
  formatted_tags = ", ".join(f'"{t}"' for t in clean_tags if t)

  markdown_template = f"""---
category: {clean_folder}
tags: [{formatted_tags}]
---
# {clean_filename.replace('.md', '').replace('-', ' ').title()}

{content.strip()}
"""
  with open(file_path, "w", encoding="utf-8") as f:
    f.write(markdown_template)

  return f"Success: Note saved successfully to {file_path}"


def list_context_files() -> str:
  """Lists all available markdown files in the local knowledge base."""
  if not BASE_DIR.exists():
    return "Directory ./my_knowledge_base is empty or does not exist."

  files = [str(f.relative_to(BASE_DIR)) for f in BASE_DIR.glob("**/*.md")]
  if not files:
    return "No markdown files found in ./my_knowledge_base."

  return "Available markdown files:\n" + "\n".join(files)


def read_context_file(relative_path: str) -> str:
  """Reads the text content of a specific local markdown file."""
  target_file = (BASE_DIR / relative_path).resolve()

  # Path traversal protection
  if not str(target_file).startswith(str(BASE_DIR.resolve())):
    return "Error: Access outside the base directory is forbidden."

  if not target_file.exists():
    return f"Error: File '{relative_path}' not found."

  return target_file.read_text(encoding="utf-8")


# JSON Schemas for LM Studio
tools = [
    {
        "type": "function",
        "function": {
            "name": "save_note_to_disk",
            "description": (
                "Saves a learning concept to disk. Only invoke when the user"
                " explicitly requests to save/log/archive."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "filename": {"type": "string"},
                    "folder": {"type": "string"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["content", "filename", "folder", "tags"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_context_files",
            "description": (
                "Lists all available reference markdown files in the local"
                " knowledge base."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_context_file",
            "description": "Reads the content of a local markdown file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {
                        "type": "string",
                        "description": (
                            "Relative file path (e.g.,"
                            " 'Python/list-comprehensions.md')."
                        )
                    }
                },
                "required": ["relative_path"],
            },
        },
    },
]

available_functions = {
    "save_note_to_disk": save_note_to_disk,
    "list_context_files": list_context_files,
    "read_context_file": read_context_file,
}


# ---------------------------------------------------------
# 2. Tool Execution Helper
# ---------------------------------------------------------
def process_agent_turn(client: OpenAI, messages: list, tool_choice="auto"):
  """Executes LLM request and automatically resolves function calls until completion."""
  while True:
    response = client.chat.completions.create(
        model="local-model",
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=0.7,
    )
    response_msg = response.choices[0].message
    messages.append(response_msg)

    # If the model requests function calls, execute them and feed back to conversation history
    if response_msg.tool_calls:
      for tool_call in response_msg.tool_calls:
        func_name = tool_call.function.name
        func_to_call = available_functions.get(func_name)
        func_args = json.loads(tool_call.function.arguments)

        print(f"🤖 [Agent Calling Tool]: {func_name}({func_args})")
        if func_to_call:
          tool_output = func_to_call(**func_args)
          messages.append({
              "tool_call_id": tool_call.id,
              "role": "tool",
              "name": func_name,
              "content": tool_output,
          })
      # Switch tool_choice back to auto after completing forced initial turn
      tool_choice = "auto"
    else:
      return response_msg.content


# ---------------------------------------------------------
# 3. Main Chat Loop
# ---------------------------------------------------------
def run_chat_loop():
  client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

  system_instruction = (
      "You are a personal learning assistant. Your goal is to explain concepts"
      " clearly. "
      "You have access to tools for browsing, reading, and saving local markdown"
      " files. "
      "When asked to load the knowledge base, inspect available files using"
      " `list_context_files` "
      "and retrieve their contents using `read_context_file` so you can keep"
      " them in memory."
  )

  messages = [{"role": "system", "content": system_instruction}]

  print("=" * 60)
  print("🎓 Personal Knowledge Assistant Initialized (LM Studio)")
  print("=" * 60)

  # Interactive prompt to load files prior to main session
  load_kb = (
      input("\nWould you like to load the local 'knowledge base' into memory?"
            " (yes/no): ")
      .strip()
      .lower()
  )

  if load_kb in ["yes", "y"]:
    print("\n[Agent]: Discovering and indexing local files into memory...")
    messages.append({
        "role": "user",
        "content": (
            "Please list all files in the knowledge base and read all of their"
            " contents into memory so you have context for our conversation."
        ),
    })

    # Force the agent to execute list_context_files first
    agent_ack = process_agent_turn(
        client,
        messages,
        tool_choice={
            "type": "function",
            "function": {"name": "list_context_files"},
        },
    )
    print(f"\nAI: {agent_ack}")

  print("\n" + "-" * 60)
  print("Ready! Ask questions normally or tell the AI to save insights.")
  print("Type 'exit' or 'quit' to end session.")
  print("-" * 60)

  while True:
    try:
      user_input = input("\nYou: ").strip()
      if not user_input:
        continue
      if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

      messages.append({"role": "user", "content": user_input})
      ai_response = process_agent_turn(client, messages, tool_choice="auto")
      print(f"\nAI: {ai_response}")

    except Exception as e:
      print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
  run_chat_loop()