from openai import OpenAI
import json

@staticmethod
def process_agent_turn(client: OpenAI, 
                       messages: list, 
                       tools:list, 
                       available_functions:dict, 
                       tool_choice="auto"):
  """Handles LLM calls and executes tool chaining sequentially until the model produces a text response."""
  current_tool_choice = tool_choice

  while True:
    response = client.chat.completions.create(
        model="local-model",
        messages=messages,
        tools=tools,
        tool_choice=current_tool_choice,
        temperature=0.7,
        parallel_tool_calls=False,
    )

    current_tool_choice = "auto"
    response_msg = response.choices[0].message

    if response_msg.tool_calls:
      # Append serialized dict version of assistant's tool call message
      tool_calls_dict = [
          tc.model_dump() if hasattr(tc, "model_dump") else tc
          for tc in response_msg.tool_calls
      ]
      messages.append({
          "role": "assistant",
          "content": response_msg.content,
          "tool_calls": tool_calls_dict,
      })

      for tool_call in response_msg.tool_calls:
        func_name = tool_call.function.name
        func_to_call = available_functions.get(func_name)

        try:
          func_args = json.loads(tool_call.function.arguments)
        except Exception:
          func_args = {}

        print(f"🤖 [Agent Executing Tool]: {func_name}({func_args})")

        if func_to_call:
          tool_output = func_to_call(**func_args)
        else:
          tool_output = f"Error: Function '{func_name}' is not recognized."

        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": func_name,
            "content": str(tool_output),
        })
    else:
      messages.append({"role": "assistant", "content": response_msg.content})
      return response_msg.content