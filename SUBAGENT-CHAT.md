# Goose Subagent Chat

A platform extension that gives concurrent subagents a shared message bus for coordination. When multiple `delegate` tasks run in parallel, they can use chat tools to announce intentions, claim resources, and avoid conflicts.

## Quick Start

```bash
# Build
cargo build -p goose-cli

# Enable the chat extension (one-time)
# In goose, run: /extensions → enable "chat"

# Then use delegate with multiple async tasks — they'll have chat tools available
```

## How It Works

```
┌──────────────────────────────────┐
│          Parent Agent            │
│   delegate(async) × N            │
└───┬─────┬──────┬──────┬──────┬───┘
    │     │      │      │      │
  ┌─▼─┐ ┌─▼─┐  ┌─▼─┐  ┌─▼─┐  ┌─▼─┐
  │ 1 │ │ 2 │  │ 3 │  │ 4 │  │ 5 │   (subagents)
  └─┬─┘ └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘
    │     │      │      │      │
  ┌─▼─────▼──────▼──────▼──────▼─────┐
  │       Global ChatBus             │
  │ in-memory message store          │
  │ atomic claims (compare-and-swap) │
  │ transcript → chat_transcript.md  │
  └──────────────────────────────────┘
```

Each subagent gets a `ChatClient` platform extension that connects to the same global `ChatBus` singleton. Subagents are identified by a stable truncated hash of their session ID (e.g., `agent-a3f2c1`).

### Chat Tools

| Tool | Description |
|------|-------------|
| `chat_claim(resource)` | Atomically claim a resource (file, number, task). First caller wins; others get an **error** with the owner's name. Use simple canonical names (e.g. `3` not `number 3`). |
| `chat_release(resource)` | Release a previously claimed resource. Only the owner can release. |
| `chat_send(message)` | Broadcast a message to all other subagents. |
| `chat_read()` | Read new messages since last read. Returns your identity in the header. |
| `chat_read(all=true)` | Read the full transcript from the beginning. |
| `chat_list()` | List all registered subagents. |

### Identity

Each subagent's identity is a truncated hash of its session ID (e.g., `agent-a3f2c1`). The `chat_read` response includes an identity header:

```
(your identity: agent-a3f2c1)
[agent-7b9e04] CLAIMED: 3
[agent-d12f88] I'm working on the auth module
```

### Transcript

All messages are appended to `chat_transcript.md` in the project working directory. The file is created on first message and includes timestamps.

## Architecture

### Global Singleton

The `ChatBus` is a process-wide singleton (`static Lazy<ChatBus>`), shared across all subagents. This avoids the complexity of wiring shared state through the extension config system, which uses plain function pointers for client factories.

### Platform Extension

`ChatClient` implements `McpClientTrait` and is registered in `PLATFORM_EXTENSIONS` with `default_enabled: false`. When enabled, subagents spawned via `delegate` inherit it automatically.

### Atomic Claims

`chat_claim` uses a `HashMap` behind a `Mutex` with entry-based insert-or-fail semantics. The first agent to claim a resource wins; all subsequent attempts return a `CallToolResult::error` with the owner's identity. This makes denied claims clearly visible to the LLM as tool failures.

Note: `chat_release` allows reclaiming resources. For tasks where each agent should pick a *unique* resource, agents should **not** release their claims until all work is complete.

### Notification Suppression

Chat tool invocations (`chat_send`, `chat_read`, `chat_claim`, `chat_release`, `chat_list`) are filtered from the CLI's subagent notification display to reduce noise.

### Known Issue: Notification Duplication

When N delegate calls are in flight simultaneously, each creates a subscriber on the shared `notification_subscribers` list. Any subagent notification is broadcast to all N subscribers, causing each tool call to be displayed N times. This is a pre-existing issue in the summon notification bridge architecture, not specific to the chat extension.

## Files

| File | Purpose |
|------|---------|
| `crates/goose/src/agents/platform_extensions/chat.rs` | ChatBus + ChatClient platform extension |
| `crates/goose/src/agents/platform_extensions/mod.rs` | Registration in PLATFORM_EXTENSIONS |
| `crates/goose-cli/src/session/mod.rs` | Notification suppression for chat tools |
| `crates/goose/src/agents/platform_extensions/summon.rs` | Model config fix for subagent sessions |

## Example Usage

Ask goose to delegate parallel tasks with coordination:

> Spin up 10 async delegates to each write a unique number 1-10 to work.txt via ./work.sh. Have them use chat_claim to claim their number before starting, so no two agents work on the same number.

The subagents will:
1. Call `chat_claim("3")` → success or error
2. If error, try `chat_claim("7")` → etc.
3. Call `./work.sh 3` to do the work
4. Call `chat_send("Finished 3")` when done

Check `chat_transcript.md` afterwards for the full coordination log.
