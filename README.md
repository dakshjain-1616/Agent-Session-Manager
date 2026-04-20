# Agent Session Manager

> Made Autonomously Using [NEO - Your Autonomous AI Engineering Agent](https://heyneo.com)
>
> [![VS Code Extension](https://img.shields.io/badge/VS%20Code-NEO%20Extension-blue?logo=visualstudiocode)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)  [![Cursor Extension](https://img.shields.io/badge/Cursor-NEO%20Extension-purple?logo=cursor)](https://marketplace.cursorapi.com/items/?itemName=NeoResearchInc.heyneo)

A lightweight Python library that gives your agents durable memory. Sessions, decisions, tool outputs, and error history persist across runs — restored in under five seconds, with no cloud dependency and no API keys.

![Architecture](architecture.svg)

## The Problem

The single biggest pain in production agents is that large language models have no memory of their own. Close the process, restart the script, or hit a token limit mid-run and everything the agent worked out — the plan it formed, the constraints it discovered, the tools it already called, the mistakes it already corrected — vanishes. The next run starts from zero.

Teams paper over this in three ways, and all three break:

1. **Stuff everything back into the prompt.** Works until the context window fills up. Then the agent starts forgetting the beginning of its own conversation and the cost of every call balloons.
2. **Write custom JSON files per project.** Works for a weekend. By the time you have multiple agents, revision history, and tool output caching, you have reinvented a database badly.
3. **Reach for a hosted "agent memory" SaaS.** Adds a dependency, a bill, and a network round-trip to the critical path — plus you have now shipped your agent's state to a third party.

This library is what sits underneath an agent when none of those three options are acceptable.

## What It Gives You

**State persistence.** Every session has a goal, a list of completed steps with their results, a queue of pending steps, an error history, a cache of tool outputs, and arbitrary metadata. All of it is stored in a local SQLite database, so a single file captures the full state of every agent run you have ever done.

**Semantic recall.** The agent can ask "what did we decide about the auth module" and get back the relevant past context — not by exact-key lookup but by meaning, using local sentence-transformer embeddings indexed in a local ChromaDB store. Past decisions stop disappearing into a log nobody reads.

**Context budget enforcement.** When you restore a long-running session into a fresh prompt, the library trims intelligently: it keeps the goal, the most recent steps, and the items the semantic layer flags as relevant, dropping older low-value history until you are under the token ceiling you specified. The agent gets the context that matters without blowing its window.

**Multi-agent support.** Several agents can share a session and see each other's decisions, or keep their memories isolated, or selectively recall across agents — a research agent can look up what the code agent decided without either leaking their full state.

**No cloud, no keys.** SQLite and ChromaDB run locally. The embedding model runs locally. Your agent state stays on your machine.

## Why This Shape

Every design choice here exists to make the library drop-in for an existing agent. There is no orchestration layer to rewrite around. The `SessionManager` is a plain Python object that you instantiate once and call from inside whatever agent loop you already have. State shape is a dataclass, not a framework abstraction — so the same library fits a LangChain agent, a LlamaIndex workflow, a raw API loop, or anything else.

The SQLite layer handles structured facts (what happened, when, with what result). The ChromaDB layer handles fuzzy retrieval (what does this situation remind me of). Splitting those two concerns means the library answers both "give me the last five tool calls" and "have we seen something like this before" without one subsystem distorting the other.

The five-second restore target is not arbitrary. It is the threshold below which developers stop working around the system. Anything slower and people start caching things themselves, and then you have the scattered-JSON problem all over again.

## When to Reach For It

You want this library if your agent runs for more than one invocation — if it resumes work the next morning, picks up after a crash, hands off to a sibling agent, or just needs to avoid re-deriving facts it already knows. You do **not** need this library for a stateless one-shot script that answers a single question and exits.

It is also a good fit for evaluation harnesses and benchmarks, where you want to inspect exactly what an agent did after the run is over.

## Running It

Install the dependencies from `requirements.txt`, point the `SessionManager` at a directory for its SQLite and ChromaDB files, and call `create_session` / `load_session` / `add_step` / `recall_context` from inside your agent.

Three example scripts in the repository walk through the typical use cases end to end:

- `examples/basic_usage.py` — single agent, persistence across runs, semantic recall
- `examples/multi_agent.py` — multiple agents sharing and isolating context
- `demo.py` — full feature tour including context-budget trimming

## Tech Stack

Python 3.10+, SQLite via the standard library, ChromaDB for the vector store, `sentence-transformers` for local embeddings (default model `all-MiniLM-L6-v2` — small, fast, good enough for recall), and `tiktoken` for token accounting.

## License

MIT.
