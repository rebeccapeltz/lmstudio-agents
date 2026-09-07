import ast
import re
from pathlib import Path

KB_DIR = Path("./my_knowledge_base").resolve() # root directory for knowledge base files

@staticmethod
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

#     target_dir = folder / clean_folder
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

# def save_note_to_disk(
#     content: str, filename: str, folder: str, tags: list[str] | str
# ) -> str:
#   """Saves a note into a specific subfolder inside the knowledge base directory."""
#   try:
#     # 1. Sanitize the subfolder name provided by the model (e.g., 'test')
#     clean_subfolder = re.sub(r"[^\w\s-]", "", folder).strip()
#     clean_filename = re.sub(r"[^\w\s.-]", "", filename).strip()
#     if not clean_filename.endswith(".md"):
#       clean_filename += ".md"

#     # 2. Join KB_DIR with the clean subfolder name
#     # e.g., Path("./my_knowledge_base") / "test"
#     target_dir = (KB_DIR / clean_subfolder).resolve()

#     # 3. Security check: Ensure target path remains within KB_DIR
#     if not str(target_dir).startswith(str(KB_DIR.resolve())):
#       return "Error: Cannot save outside the designated knowledge base folder."

#     target_dir.mkdir(parents=True, exist_ok=True)
#     file_path = target_dir / clean_filename

#     # ... frontmatter tag formatting & file writing logic ...

#     with open(file_path, "w", encoding="utf-8") as f:
#       f.write(markdown_template)

#     return f"Success: Note saved successfully to {file_path}"
#   except Exception as e:
#     return f"Error saving file: {str(e)}"




def save_note_to_disk(
    content: str, filename: str, folder: str, tags: list[str] | str
) -> str:
  """Saves a note into a specific subfolder inside the knowledge base directory."""
  try:
    # 1. Normalize and clean tags (handles both list and stringified list formats)
    if isinstance(tags, str):
      tags_str = tags.strip()
      if tags_str.startswith("[") and tags_str.endswith("]"):
        try:
          parsed = ast.literal_eval(tags_str)
          tags = parsed if isinstance(parsed, list) else [tags_str]
        except Exception:
          tags = [
              t.strip() for t in tags_str.strip("[]").split(",") if t.strip()
          ]
      else:
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]

    clean_tags = [re.sub(r"[^\w\s-]", "", str(tag)).strip() for tag in tags]
    formatted_tags = ", ".join(f'"{t}"' for t in clean_tags if t)

    # 2. Sanitize subfolder and file names
    clean_subfolder = re.sub(r"[^\w\s-]", "", folder).strip()
    clean_filename = re.sub(r"[^\w\s.-]", "", filename).strip()
    if not clean_filename.endswith(".md"):
      clean_filename += ".md"

    # 3. Resolve target path relative to KB_DIR
    target_dir = (KB_DIR / clean_subfolder).resolve()

    # Security check: Ensure target path remains within KB_DIR
    if not str(target_dir).startswith(str(KB_DIR)):
      return "Error: Cannot save outside the designated knowledge base folder."

    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / clean_filename

    # 4. Format Frontmatter with tags included
    markdown_template = f"""---
category: {clean_subfolder}
tags: [{formatted_tags}]
---
# {clean_filename.replace('.md', '').replace('-', ' ').title()}

{content.strip()}
"""

    with open(file_path, "w", encoding="utf-8") as f:
      f.write(markdown_template)

    return f"Success: Note saved successfully to {file_path}"
  except Exception as e:
    return f"Error saving file: {str(e)}"

tools = [{
    "type": "function",
    "function": {
        "name": "save_note_to_disk",
        "description": (
            "Saves a specific learning concept or AI answer to a local markdown"
            " file. Only invoke when explicitly requested by user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "The text or concept explanation to be saved."
                    ),
                },
                "filename": {
                    "type": "string",
                    "description": (
                        "A short, clean file name ending in .md (e.g.,"
                        " 'list-comprehensions.md')."
                    ),
                },
                "folder": {
                    "type": "string",
                    "description": (
                        "The subject folder category (e.g., 'Python',"
                        " 'Machine Learning'). Do not include"
                        " root directory paths."
                    ),
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "A list of 2-4 keywords for future organization."
                    ),
                },
            },
            "required": ["content", "filename", "folder", "tags"],
        },
    },
}]

available_functions = {
    "save_note_to_disk": save_note_to_disk,
}
