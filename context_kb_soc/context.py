
@staticmethod
def load_context(CONTEXT_DIR):
    context_content_list = []
    context_str = ""
    for md_file in CONTEXT_DIR.glob("**/*.md"):
        try:
            rel_path = md_file.relative_to(CONTEXT_DIR)
            text = md_file.read_text(encoding="utf-8")
            context_content_list.append(f"--- FILE: {rel_path} ---\n{text}\n")
        except Exception as e:
            print(f"Error reading {md_file}: {e}")
    if context_content_list:
          context_str = "\n".join(context_content_list)
    return context_str