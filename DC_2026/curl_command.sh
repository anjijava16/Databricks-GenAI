welcome@jaisairams-Laptop mcp_servers % curl https://dbc-d1d2ef51-1174.cloud.databricks.com/serving-endpoints/databricks-gpt-5-4/invocations \
  -X POST \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
  "messages": [
    {
      "role": "user",
      "content": "What is an LLM agent?"
    }
  ],
  "max_tokens": 5000
}'
{
  "id": "chatcmpl-DPFpYtO9OD2QFA9Dsc58f59pHPJ3B",
  "object": "chat.completion",
  "created": 1774911424,
  "model": "gpt-5.4-2026-03-05",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "An **LLM agent** is a system built around a **large language model (LLM)** that can do more than just generate text. It can:\n\n- **reason about a task**\n- **decide what actions to take**\n- **use tools or APIs**\n- **observe results**\n- **continue until it reaches a goal**\n\nSo instead of only answering a prompt once, an agent acts more like a loop:\n\n1. **Receive a goal**  \n   “Book a flight,” “summarize these files,” “debug this app,” etc.\n\n2. **Plan or choose a next step**  \n   It figures out what to do first.\n\n3. **Use tools**  \n   Examples:\n   - web search\n   - calculators\n   - databases\n   - code execution\n   - email/calendar APIs\n   - file systems\n\n4. **Read the result**\n   It checks what happened.\n\n5. **Repeat if needed**\n   It keeps going until the task is done or it gets stuck.\n\n## Simple way to think about it\n\n- **Plain LLM**: “Here’s an answer.”\n- **LLM agent**: “Here’s what I’ll do, I’ll use these tools, check the results, and then decide the next step.”\n\n## Example\n\nIf you ask:\n\n> “Find the cheapest nonstop flight to Chicago next Friday and put the options in a table.”\n\nA normal LLM might only describe how to do that.\n\nAn **LLM agent** could:\n- search flight sites via tools\n- compare options\n- format the results into a table\n- maybe even save it to a file or send it to you\n\n## Common components of an LLM agent\n\n- **LLM**: the reasoning and language engine\n- **Memory**: stores conversation or past steps\n- **Tools**: external functions, APIs, software\n- **Planner/controller**: manages the action loop\n- **Goal/state**: what it’s trying to achieve and current progress\n\n## Important note\n\n“Agent” can mean different things in different contexts. Some so-called agents are just:\n- an LLM with function calling\n- an LLM with a workflow\n- a more autonomous system with planning and memory\n\nSo the term is a bit broad.\n\nIf you want, I can also explain:\n- **LLM agent vs chatbot**\n- **LLM agent vs workflow automation**\n- **how agents are built in practice**",
        "refusal": null,
        "annotations": []
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 13,
    "completion_tokens": 504,
    "total_tokens": 517,
    "prompt_tokens_details": {
      "cached_tokens": 0,
      "audio_tokens": 0
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0,
      "audio_tokens": 0,
      "accepted_prediction_tokens": 0,
      "rejected_prediction_tokens": 0
    }
  },
  "service_tier": "default",
  "system_fingerprint": null
}
welcome@jaisairams-Laptop mcp_servers % 