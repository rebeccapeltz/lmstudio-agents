# import ast
# import json
# import re
from pathlib import Path
from openai import OpenAI

import tool
import tool_execution
import context


# ---------------------------------------------------------
# 1. Base Configuration & Tool Implementations
# ---------------------------------------------------------
KB_DIR = Path("./my_knowledge_base").resolve()  # root directory for knowledge base files
CONTEXT_DIR = Path("./context_files").resolve()  # root directory for context files


# def save_note_to_disk(
#     content: str, filename: str, folder: str, tags: list[str] | str
# ) -> str:
#   """Saves a specific learning concept or AI answer to a local markdown file."""
#   try:
#     if isinstance(tags, str):
#       tags_str = tags.strip()
#       if tags_str.startswith("[") and tags_str.endswith("]"):
#         try:
#           parsed = ast.literal_eval(tags_str)
#           tags = parsed if isinstance(parsed, list) else [tags_str]
#         except Exception:
#           tags = [
#               t.strip() for t in tags_str.strip("[]").split(",") if t.strip()
#           ]
#       else:
#         tags = [t.strip() for t in tags_str.split(",") if t.strip()]

#     clean_folder = re.sub(r"[^\w\s-]", "", folder).strip()
#     clean_filename = re.sub(r"[^\w\s.-]", "", filename).strip()
#     if not clean_filename.endswith(".md"):
#       clean_filename += ".md"

#     target_dir = KB_DIR / clean_folder
#     target_dir.mkdir(parents=True, exist_ok=True)
#     file_path = target_dir / clean_filename

#     clean_tags = [re.sub(r"[^\w\s-]", "", str(tag)).strip() for tag in tags]
#     formatted_tags = ", ".join(f'"{t}"' for t in clean_tags if t)

#     markdown_template = f"""---
# category: {clean_folder}
# tags: [{formatted_tags}]
# ---
# # {clean_filename.replace('.md', '').replace('-', ' ').title()}

# {content.strip()}
# """
#     with open(file_path, "w", encoding="utf-8") as f:
#       f.write(markdown_template)

#     return f"Success: Note saved successfully to {file_path}"
#   except Exception as e:
#     return f"Error saving file: {str(e)}"


# tools = [{
#     "type": "function",
#     "function": {
#         "name": "save_note_to_disk",
#         "description": (
#             "Saves a specific learning concept or AI answer to a local markdown"
#             " file. Only invoke when explicitly requested by user."
#         ),
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "content": {
#                     "type": "string",
#                     "description": (
#                         "The text or concept explanation to be saved."
#                     ),
#                 },
#                 "filename": {
#                     "type": "string",
#                     "description": (
#                         "A short, clean file name ending in .md (e.g.,"
#                         " 'list-comprehensions.md')."
#                     ),
#                 },
#                 "folder": {
#                     "type": "string",
#                     "description": (
#                         "The subject folder category (e.g., 'Python',"
#                         " 'Machine Learning')."
#                     ),
#                 },
#                 "tags": {
#                     "type": "array",
#                     "items": {"type": "string"},
#                     "description": (
#                         "A list of 2-4 keywords for future organization."
#                     ),
#                 },
#             },
#             "required": ["content", "filename", "folder", "tags"],
#         },
#     },
# }]

# available_functions = {
#     "save_note_to_disk": save_note_to_disk,
# }


# ---------------------------------------------------------
# 2. Execution Engine
# ---------------------------------------------------------
# def process_agent_turn(client: OpenAI, messages: list,tool.tools: list, tool.available_functions: dict, tool_choice="auto"):
#   """Handles LLM calls and executes tool chaining sequentially until the model produces a text response."""
#   current_tool_choice = tool_choice

#   while True:
#     response = client.chat.completions.create(
#         model="local-model",
#         messages=messages,
#         tools=tool.tools,
#         tool_choice=current_tool_choice,
#         temperature=0.7,
#         parallel_tool_calls=False,
#     )

#     current_tool_choice = "auto"
#     response_msg = response.choices[0].message

#     if response_msg.tool_calls:
#       # Append serialized dict version of assistant's tool call message
#       tool_calls_dict = [
#           tc.model_dump() if hasattr(tc, "model_dump") else tc
#           for tc in response_msg.tool_calls
#       ]
#       messages.append({
#           "role": "assistant",
#           "content": response_msg.content,
#           "tool_calls": tool_calls_dict,
#       })

#       for tool_call in response_msg.tool_calls:
#         func_name = tool_call.function.name
#         func_to_call = available_functions.get(func_name)

#         try:
#           func_args = json.loads(tool_call.function.arguments)
#         except Exception:
#           func_args = {}

#         print(f"🤖 [Agent Executing Tool]: {func_name}({func_args})")

#         if func_to_call:
#           tool_output = func_to_call(**func_args)
#         else:
#           tool_output = f"Error: Function '{func_name}' is not recognized."

#         messages.append({
#             "tool_call_id": tool_call.id,
#             "role": "tool",
#             "name": func_name,
#             "content": str(tool_output),
#         })
#     else:
#       messages.append({"role": "assistant", "content": response_msg.content})
#       return response_msg.content


# ---------------------------------------------------------
# 3. Main Interactive Chat Session
# ---------------------------------------------------------
def run_chat_loop():
  client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

  system_instruction = (
      "You are a personal learning assistant. Your goal is to explain concepts"
      " clearly. "
      "You have access to a tool (`tool.save_note_to_disk`) that saves markdown"
      " notes on disk. "
      "NEVER call `toolsave_note_to_disk` unless the user explicitly requests to"
      " save, log, or archive."
  )

  messages = [{"role": "system", "content": system_instruction}]

  print("=" * 60)
  print("🎓 Personal Knowledge Assistant (LM Studio)")
  print("=" * 60)

  # Ensure context directory exists
  CONTEXT_DIR.mkdir(parents=True, exist_ok=True)

  load_kb = (
      input("\nWould you like to load local context files into memory?"
            " (yes/no): ")
      .strip()
      .lower()
  )

  if load_kb in ["yes", "y"]:
    print("\n[System]: Reading context files into memory...")
    context_string = context.load_context(CONTEXT_DIR)    
    # kb_content = []

    # for md_file in CONTEXT_DIR.glob("**/*.md"):
    #   try:
    #     rel_path = md_file.relative_to(CONTEXT_DIR)
    #     text = md_file.read_text(encoding="utf-8")
    #     kb_content.append(f"--- FILE: {rel_path} ---\n{text}\n")
    #   except Exception as e:
    #     print(f"Error reading {md_file}: {e}")

    # if kb_content:
    # if context_content.length > 0:
    #   full_kb_str = "\n".join(context_content)
    if context_string:
      messages.append({
          "role": "user",
          "content": (
              "Here is my reference context for our session:\n\n"
              f"{context_string}"
          ),
      })
      messages.append({
          "role": "assistant",
          "content": (
              "Thank you. I have loaded all context files into memory and am"
              " ready to answer your questions using this information."
          ),
      })
      print("✅ Context loaded successfully.")
    else:
      print("⚠️ No markdown files found in ./context_files to load.")

  print("\n" + "-" * 60)
  print("Ready! Ask questions normally or request to save insights.")
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
      ai_response = tool_execution.process_agent_turn(client, 
                                                      messages, 
                                                      tool.tools, 
                                                      tool.available_functions, 
                                                      tool_choice="auto")
      print(f"\nAI: {ai_response}")

    except Exception as e:
      print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
  run_chat_loop()