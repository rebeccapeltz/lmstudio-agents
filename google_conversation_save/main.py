import os
import re
from pathlib import Path
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. Define the Local File-Saving Tool
# ---------------------------------------------------------
def save_note_to_disk(content: str, filename: str, folder: str, tags: list[str]) -> str:
    """
    Saves a specific learning concept or AI answer to a local markdown file.
    
    Args:
        content: The text or concept explanation to be saved.
        filename: A short, clean file name ending in .md (e.g., 'list-comprehensions.md').
        folder: The subject folder category (e.g., 'Python', 'Machine Learning', 'History').
        tags: A list of 2-4 keywords for future organization.
    """
    # Sanitize inputs to prevent path traversal or bad filenames
    clean_folder = re.sub(r'[^\w\s-]', '', folder).strip()
    clean_filename = re.sub(r'[^\w\s.-]', '', filename).strip()
    if not clean_filename.endswith('.md'):
        clean_filename += '.md'
        
    # Define your local notes root directory
    base_dir = Path("./my_knowledge_base")
    target_dir = base_dir / clean_folder
    target_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = target_dir / clean_filename
    
    # Format the file with clean Markdown Frontmatter
    formatted_tags = ", ".join(tags)
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


# ---------------------------------------------------------
# 2. Main Chat Loop Initialization
# ---------------------------------------------------------
def run_chat_loop():
    # Initialize the client (automatically picks up GEMINI_API_KEY env variable)
    client = genai.Client()
    
    # Use gemini-2.5-flash as the fast, intelligent default model
    model_id = "gemini-2.5-flash"
    
    # System instructions guide the model's behavior and guardrails
    system_instruction = (
        "You are a personal learning assistant. Your goal is to explain concepts clearly. "
        "You have access to a tool called `save_note_to_disk`. NEVER call this tool automatically. "
        "Only call this tool when the user explicitly requests to save, archive, or log information "
        "to their disk. When saving, automatically deduce a concise, clean filename, target folder, "
        "and relevant tags based on the user's intent and the context of the conversation."
    )
    
    # Provide the tool directly to the chat configuration
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[save_note_to_disk], 
        temperature=0.7
    )
    
    # Create the persistent chat session
    chat = client.chats.create(model=model_id, config=config)
    
    print("=" * 60)
    print("🎓 Personal Knowledge Assistant Initialized.")
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
                
            # Send message to the model. The SDK handles tool execution automatically.
            response = chat.send_message(user_input)
            
            print(f"\nAI: {response.text}")
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    run_chat_loop()