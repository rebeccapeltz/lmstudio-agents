"""
Minimal local agent against LM Studio's OpenAI-compatible endpoint.

Architecture recap (see chat for full explanation):
  - LM Studio only does inference. It never touches the network or the disk.
  - THIS SCRIPT is what makes outbound HTTP requests and local file writes,
    on the model's behalf, whenever the model asks to call a tool.

Requirements:
  pip install requests

Before running:
  1. Open LM Studio, load a tool-calling-capable model (check the model card —
     not all HF models are fine-tuned for tool use).
  2. Start the local server (LM Studio -> Developer -> Start Server).
     Default endpoint: http://localhost:1234/v1
"""

import json
import os
import requests

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "local-model"  # LM Studio ignores this or uses whichever is loaded

# Sandbox directory for write_file. The agent can ONLY write inside here --
# this is the guardrail that keeps a hallucinated or injected path from
# touching the rest of your filesystem. Created on first write if missing.
SANDBOX_DIR = os.path.join(os.getcwd(), "agent_workspace")

# ---------------------------------------------------------------------------
# Step 1: Define the tool(s) available to the model, in OpenAI tool-call format
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch the raw text content of a public URL. No authentication "
                "is required or supported. Use this to read web pages, RSS "
                "feeds, or public APIs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to fetch, including https://",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write text content to a file. The path is relative and is "
                "always confined to a local sandbox folder (agent_workspace) "
                "-- absolute paths and '..' are rejected. Use this to save "
                "notes, summaries, or fetched content for the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": (
                            "Relative filename only, e.g. 'summary.txt'. "
                            "No slashes, no '..', no absolute paths."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write to the file.",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Step 2: Implement the actual tool. This is where the real network call happens.
# ---------------------------------------------------------------------------

def fetch_url(url: str, max_chars: int = 3000) -> str:
    """Perform a no-auth GET request and return trimmed text content.

    This function -- not the model, not LM Studio -- is what talks to the
    internet. Treat it as untrusted input: a fetched page could contain text
    designed to manipulate the agent (prompt injection). We don't execute
    anything from the fetched content, only pass it back as text for the
    model to read.
    """
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "local-agent-demo/0.1"},
            timeout=10,
        )
        resp.raise_for_status()
        text = resp.text
        return text[:max_chars]
    except requests.RequestException as e:
        return f"[fetch_url error] {e}"


def write_file(filename: str, content: str) -> str:
    """Write content to a file, confined to SANDBOX_DIR.

    This is the guardrail step: we resolve the final path and verify it is
    still inside the sandbox before writing anything. This blocks both
    obvious attempts ("../../.bashrc") and sneakier absolute-path attempts,
    whether they come from a model mistake or from injected instructions in
    fetched web content.
    """
    # Reject anything that isn't a plain filename up front.
    if os.path.isabs(filename) or ".." in filename.replace("\\", "/").split("/"):
        return f"[write_file error] rejected unsafe path: {filename!r}"

    os.makedirs(SANDBOX_DIR, exist_ok=True)
    target = os.path.abspath(os.path.join(SANDBOX_DIR, filename))
    sandbox_root = os.path.abspath(SANDBOX_DIR)

    # Belt-and-suspenders: confirm the resolved path is still inside the sandbox.
    if os.path.commonpath([target, sandbox_root]) != sandbox_root:
        return f"[write_file error] rejected path outside sandbox: {filename!r}"

    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[write_file ok] wrote {len(content)} chars to {target}"
    except OSError as e:
        return f"[write_file error] {e}"


TOOL_IMPLEMENTATIONS = {
    "fetch_url": fetch_url,
    "write_file": write_file,
}


# ---------------------------------------------------------------------------
# Step 3: The agent loop itself
# ---------------------------------------------------------------------------

def call_lm_studio(messages, tools=None):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
    }
    if tools:
        payload["tools"] = tools

    resp = requests.post(LM_STUDIO_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def run_agent(user_prompt: str, max_turns: int = 5):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a local research assistant. You have a fetch_url tool "
                "for reading public web pages (no credentials available or "
                "needed) and a write_file tool for saving results to a local "
                "sandbox folder. Use fetch_url when you need current "
                "information, and write_file when the user asks you to save "
                "or record something."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    for turn in range(max_turns):
        result = call_lm_studio(messages, tools=TOOLS)
        choice = result["choices"][0]
        message = choice["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            # No tool call -> model gave a final answer
            return message.get("content", "")

        # Model asked for one or more tool calls -- execute each locally
        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args = json.loads(call["function"]["arguments"])
            impl = TOOL_IMPLEMENTATIONS.get(fn_name)

            if impl is None:
                tool_result = f"[error] unknown tool {fn_name}"
            else:
                print(f"  -> agent is calling {fn_name}({fn_args})")
                tool_result = impl(**fn_args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": tool_result,
                }
            )
        # loop again so the model can read the tool result and respond

    return "[stopped: max_turns reached without a final answer]"


if __name__ == "__main__":
    question = (
        "What is the current top story on the Hacker News front page? "
        "Save a one-sentence summary to a file called hn_summary.txt."
    )
    print(f"User: {question}\n")
    answer = run_agent(question)
    print(f"\nAgent: {answer}")
    print(f"\n(Check the '{SANDBOX_DIR}' folder for any files written.)")
