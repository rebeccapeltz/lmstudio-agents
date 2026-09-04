import json
import re
from pathlib import Path
from openai import OpenAI

import ast

# ---------------------------------------------------------
# 1. Define the Local File-Saving Tool
# ---------------------------------------------------------

def save_note_to_disk(
    content: str, filename: str, folder: str, tags: list[str] | str
) -> str:
  """Saves a specific learning concept or AI answer to a local markdown file."""
  # Fix: Ensure tags is a proper list if the model passed it as a string
  if isinstance(tags, str):
    tags_str = tags.strip()
    if tags_str.startswith("[") and tags_str.endswith("]"):
      try:
        # Safely parse string representation of list like "['Civics', 'US History']"
        parsed = ast.literal_eval(tags_str)
        if isinstance(parsed, list):
          tags = parsed
        else:
          tags = [
              t.strip() for t in tags_str.strip("[]").split(",") if t.strip()
          ]
      except Exception:
        tags = [t.strip() for t in tags_str.strip("[]").split(",") if t.strip()]
    else:
      tags = [t.strip() for t in tags_str.split(",") if t.strip()]

  clean_folder = re.sub(r"[^\w\s-]", "", folder).strip()
  clean_filename = re.sub(r"[^\w\s.-]", "", filename).strip()
  if not clean_filename.endswith(".md"):
    clean_filename += ".md"

  base_dir = Path("./my_knowledge_base")
  target_dir = base_dir / clean_folder
  target_dir.mkdir(parents=True, exist_ok=True)

  file_path = target_dir / clean_filename

  # Clean individual tag elements and re-join nicely
  clean_tags = [re.sub(r'[^\w\s-]', '', str(tag)).strip() for tag in tags]
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

# Tool definition JSON schema for OpenAI-compatible API
tools = [
    {
        "type": "function",
        "function": {
            "name": "save_note_to_disk",
            "description": "Saves a specific learning concept or AI answer to a local markdown file. Only call when explicitly requested by user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The text or concept explanation to be saved."
                    },
                    "filename": {
                        "type": "string",
                        "description": "A short, clean file name ending in .md (e.g., 'list-comprehensions.md')."
                    },
                    "folder": {
                        "type": "string",
                        "description": "The subject folder category (e.g., 'Python', 'Machine Learning')."
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of 2-4 keywords for future organization."
                    }
                },
                "required": ["content", "filename", "folder", "tags"]
            }
        }
    }
]

# Map string tool names to local Python functions
available_functions = {
    "save_note_to_disk": save_note_to_disk
}


# ---------------------------------------------------------
# 2. Main Chat Loop Initialization
# ---------------------------------------------------------
def run_chat_loop():
    # Point client to LM Studio local server endpoint
    client = OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio"  # Any string works as api_key for local LM Studio
    )
    
    system_instruction = (
        "You are a personal learning assistant. Your goal is to explain concepts clearly. "
        "You have access to a tool called `save_note_to_disk`. NEVER call this tool automatically. "
        "Only call this tool when the user explicitly requests to save, archive, or log information "
        "to their disk. When saving, automatically deduce a concise, clean filename, target folder, "
        "and relevant tags based on the user's intent and the context of the conversation."
    )
    
    # Maintain conversational context in standard messages list
    messages = [
        {"role": "system", "content": system_instruction}
    ]
    
    print("=" * 60)
    print("🎓 Personal Knowledge Assistant Initialized (LM Studio).")
    print("Ask questions normally. To save an insight, tell the AI:")
    print("   'Save that last concept to my Python folder.'")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            messages.append({"role": "user", "content": user_input})
            
            # Request response from LM Studio
            response = client.chat.completions.create(
                model="local-model",  # LM Studio uses whichever model is currently loaded
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.7
            )
            
            response_message = response.choices[0].message
            messages.append(response_message)

            # Check if the model decided to call a function
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions.get(function_name)
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_to_call:
                        # Execute local tool
                        tool_output = function_to_call(**function_args)
                        
                        # Send tool output back to the model
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": tool_output
                        })
                        
                        # Get final summary/confirmation from model
                        second_response = client.chat.completions.create(
                            model="local-model",
                            messages=messages
                        )
                        final_content = second_response.choices[0].message.content
                        messages.append(second_response.choices[0].message)
                        print(f"\nAI: {final_content}")
            else:
                print(f"\nAI: {response_message.content}")
                
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    run_chat_loop()