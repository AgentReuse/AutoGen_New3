好的，这里是一个更简洁、面向 GitHub 开发者的 **AutoReuse** README 版本，保留技术重点，但阅读起来更轻量：

---

# AutoReuse 🚀 — Plan & Response Reuse for AutoGen MAS

## What is AutoReuse?

**AutoReuse** is an extension for [AutoGen](https://github.com/microsoft/autogen) that speeds up **Multi-Agent Systems (MAS)** by reusing **plans** and **responses** from past tasks.
Instead of rethinking everything from scratch, AutoReuse remembers *how* a task was done and *what* was said, then decides whether to reuse it or adapt it for the new request.

---

## How It Works

1. **After a Task**

   * A summarizing agent saves:

     * **Execution plan** (steps the agents took)
     * **Final response**
   * Stored in a **persistent cache**, keyed by the vectorized user input.

2. **For a New Request**

   * Vectorize the input using embedding model, we provide:
     * `3e-small`
     * `s-bert (all-MiniLM-L6-v2)`
     * `gte-small`
     * or you can choose a customized embedding model that transforms natural languages into dense vectors
   * Search the cache for similar past requests.
   * Compare similarity against two thresholds:

     * **Response Reuse Threshold** (higher)
     * **Plan Reuse Threshold** (lower)

3. **Reuse Decision**

   * **Response Reuse**: Return the cached answer directly.
   * **Plan Reuse**: Extract key parameters → Insert into saved plan → Re-run plan → Get fresh response.
   * **No Match**: Run normal MAS workflow.

---

## Quick Example

```python
from autoreuse import AutoReuseManager

reuse_manager = AutoReuseManager(
    plan_threshold=0.75,
    response_threshold=0.90,
    model="s-bert"
)

reply = reuse_manager.process_request(
    "Make a powerpoint for my German course presentation tomorrow about Porsche"
)
print(reply)
```

---

## Why Use AutoReuse?

- **Faster** responses for similar queries
- **Cheaper** by avoiding full recomputation
- **Flexible** — Plan reuse adapts to small changes in requests
- **Persistent** cache works across sessions

---

## Install

```bash
git clone https://github.com/XXXXXX
cd AutoReuse
pip install -r requirements.txt
```

