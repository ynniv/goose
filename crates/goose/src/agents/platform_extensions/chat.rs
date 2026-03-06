use crate::agents::extension::PlatformExtensionContext;
use crate::agents::mcp_client::{Error, McpClientTrait};
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use indoc::indoc;
use once_cell::sync::Lazy;
use rmcp::model::{
    CallToolResult, Content, Implementation, InitializeResult, JsonObject, ListToolsResult,
    ServerCapabilities, Tool, ToolAnnotations,
};
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

pub static EXTENSION_NAME: &str = "chat";

// ── Global ChatBus singleton ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub id: u64,
    pub from: String,
    pub to: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

struct ChatBus {
    messages: Mutex<Vec<ChatMessage>>,
    agents: Mutex<HashMap<String, String>>, // name -> role/description
    claims: Mutex<HashMap<String, String>>, // resource -> claiming agent
    next_id: AtomicU64,
    transcript_path: Mutex<Option<PathBuf>>,
}

impl ChatBus {
    fn new() -> Self {
        Self {
            messages: Mutex::new(Vec::new()),
            agents: Mutex::new(HashMap::new()),
            claims: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            transcript_path: Mutex::new(None),
        }
    }

    async fn set_transcript_path(&self, path: PathBuf) {
        let mut tp = self.transcript_path.lock().await;
        if tp.is_none() {
            // Write header on first setup
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&path)
            {
                let _ = writeln!(f, "# Chat Transcript\n");
                let _ = writeln!(f, "_Started: {}_\n", Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));
            }
            *tp = Some(path);
        }
    }

    fn append_to_transcript(path: &PathBuf, msg: &ChatMessage) {
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(path) {
            let _ = writeln!(
                f,
                "**[{}]** _{}_\n{}\n",
                msg.from,
                msg.timestamp.format("%H:%M:%S"),
                msg.content
            );
        }
    }

    async fn register_agent(&self, name: String, description: String) {
        self.agents.lock().await.entry(name).or_insert(description);
    }

    async fn send_message(&self, from: String, to: String, content: String) -> ChatMessage {
        let msg = ChatMessage {
            id: self.next_id.fetch_add(1, Ordering::SeqCst),
            from,
            to,
            content,
            timestamp: Utc::now(),
        };
        self.messages.lock().await.push(msg.clone());
        if let Some(path) = self.transcript_path.lock().await.as_ref() {
            Self::append_to_transcript(path, &msg);
        }
        msg
    }

    async fn read_messages(&self, for_agent: &str, since_id: u64) -> Vec<ChatMessage> {
        let messages = self.messages.lock().await;
        messages
            .iter()
            .filter(|m| m.id > since_id)
            .filter(|m| m.to == for_agent || m.to == "all")
            .cloned()
            .collect()
    }

    /// Atomically claim a resource. Returns Ok(()) if claimed, Err(owner) if already taken.
    async fn claim(&self, resource: String, agent: String) -> Result<(), String> {
        use std::collections::hash_map::Entry;
        let mut claims = self.claims.lock().await;
        match claims.entry(resource) {
            Entry::Vacant(e) => {
                e.insert(agent);
                Ok(())
            }
            Entry::Occupied(e) => Err(e.get().clone()),
        }
    }

    /// Release a claim. Only the owner can release it.
    async fn release(&self, resource: &str, agent: &str) -> Result<(), String> {
        let mut claims = self.claims.lock().await;
        match claims.get(resource) {
            Some(owner) if owner == agent => {
                claims.remove(resource);
                Ok(())
            }
            Some(owner) => Err(format!("'{}' is claimed by '{}', not you", resource, owner)),
            None => Err(format!("'{}' is not claimed", resource)),
        }
    }

    async fn list_agents(&self) -> Vec<(String, String)> {
        self.agents
            .lock()
            .await
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

static BUS: Lazy<ChatBus> = Lazy::new(ChatBus::new);

// ── Tool parameter schemas ────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ChatSendParams {
    /// The message to broadcast to all other subagents
    message: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ChatReadParams {
    /// If true, return the full transcript (all messages), not just new ones since last read
    #[serde(default)]
    all: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ChatClaimParams {
    /// The resource to claim (e.g. a filename, number, or task identifier)
    resource: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ChatReleaseParams {
    /// The resource to release (must have been claimed by you)
    resource: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ChatListParams {}

// ── Platform extension client ─────────────────────────────────────────

pub struct ChatClient {
    info: InitializeResult,
    context: PlatformExtensionContext,
    last_read_id: AtomicU64,
}

impl ChatClient {
    pub fn new(context: PlatformExtensionContext) -> Result<Self> {
        let info = InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new(EXTENSION_NAME.to_string(), "1.0.0".to_string())
                    .with_title("Chat"),
            )
            .with_instructions(
                indoc! {r#"
                    You are a subagent running concurrently with other subagents.
                    Use the chat tools to coordinate and avoid conflicts:
                    - chat_claim: Atomically claim a resource (file, number, task) — prevents duplicates
                    - chat_send: Broadcast a message to all other subagents
                    - chat_read: Check for messages from other subagents
                    - chat_list: See which other subagents are active

                    IMPORTANT: Use chat_claim before modifying any file or taking any
                    unique resource. If the claim is denied, pick a different resource.
                    Use chat_send/chat_read for general coordination.
                "#}
                .to_string(),
            );

        Ok(Self {
            info,
            context,
            last_read_id: AtomicU64::new(0),
        })
    }

    fn agent_name(&self, session_id: &str) -> String {
        // Truncated hash of session ID for a short, stable, unique name
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        session_id.hash(&mut hasher);
        let hash = hasher.finish();
        format!("agent-{:06x}", hash & 0xFFFFFF)
    }

    async fn ensure_registered(&self, session_id: &str) {
        let name = self.agent_name(session_id);
        let description = self
            .context
            .session
            .as_ref()
            .map(|s| s.name.clone())
            .unwrap_or_default();
        BUS.register_agent(name, description).await;

        // Set transcript path from the session's working directory
        if let Some(session) = &self.context.session {
            let path = session.working_dir.join("chat_transcript.md");
            BUS.set_transcript_path(path).await;
        }
    }

    async fn handle_send(
        &self,
        session_id: &str,
        arguments: Option<JsonObject>,
    ) -> Result<Vec<Content>, String> {
        self.ensure_registered(session_id).await;

        let message = arguments
            .as_ref()
            .ok_or("Missing arguments")?
            .get("message")
            .and_then(|v| v.as_str())
            .ok_or("Missing required parameter: message")?
            .to_string();

        if message.is_empty() {
            return Err("Message cannot be empty".to_string());
        }

        let from = self.agent_name(session_id);
        let msg = BUS.send_message(from, "all".to_string(), message).await;
        Ok(vec![Content::text(format!("Message sent (id: {})", msg.id))])
    }

    async fn handle_read(
        &self,
        session_id: &str,
        arguments: Option<JsonObject>,
    ) -> Result<Vec<Content>, String> {
        self.ensure_registered(session_id).await;

        let all = arguments
            .as_ref()
            .and_then(|a| a.get("all"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let agent_name = self.agent_name(session_id);
        let since = if all {
            0
        } else {
            self.last_read_id.load(Ordering::SeqCst)
        };
        let messages = BUS.read_messages(&agent_name, since).await;

        if let Some(last) = messages.last() {
            self.last_read_id.store(last.id, Ordering::SeqCst);
        }

        let header = format!("(your identity: {})", agent_name);

        if messages.is_empty() {
            return Ok(vec![Content::text(format!("{}\nNo new messages.", header))]);
        }

        let formatted: Vec<String> = messages
            .iter()
            .map(|m| format!("[{}] {}", m.from, m.content))
            .collect();

        Ok(vec![Content::text(format!(
            "{}\n{}",
            header,
            formatted.join("\n")
        ))])
    }

    async fn handle_claim(
        &self,
        session_id: &str,
        arguments: Option<JsonObject>,
    ) -> Result<Vec<Content>, String> {
        self.ensure_registered(session_id).await;

        let resource = arguments
            .as_ref()
            .ok_or("Missing arguments")?
            .get("resource")
            .and_then(|v| v.as_str())
            .ok_or("Missing required parameter: resource")?
            .to_string();

        let agent = self.agent_name(session_id);

        match BUS.claim(resource.clone(), agent.clone()).await {
            Ok(()) => {
                // Broadcast the successful claim
                BUS.send_message(
                    agent,
                    "all".to_string(),
                    format!("CLAIMED: {}", resource),
                )
                .await;
                Ok(vec![Content::text(format!(
                    "Successfully claimed '{}'",
                    resource
                ))])
            }
            Err(owner) => Err(format!(
                "DENIED: '{}' is already claimed by '{}'. You must pick a different resource.",
                resource, owner
            )),
        }
    }

    async fn handle_release(
        &self,
        session_id: &str,
        arguments: Option<JsonObject>,
    ) -> Result<Vec<Content>, String> {
        self.ensure_registered(session_id).await;

        let resource = arguments
            .as_ref()
            .ok_or("Missing arguments")?
            .get("resource")
            .and_then(|v| v.as_str())
            .ok_or("Missing required parameter: resource")?
            .to_string();

        let agent = self.agent_name(session_id);

        match BUS.release(&resource, &agent).await {
            Ok(()) => {
                BUS.send_message(
                    agent,
                    "all".to_string(),
                    format!("RELEASED: {}", resource),
                )
                .await;
                Ok(vec![Content::text(format!(
                    "Released '{}'",
                    resource
                ))])
            }
            Err(e) => Ok(vec![Content::text(format!("Error: {}", e))]),
        }
    }

    async fn handle_list(
        &self,
        session_id: &str,
    ) -> Result<Vec<Content>, String> {
        self.ensure_registered(session_id).await;

        let agents = BUS.list_agents().await;
        let my_name = self.agent_name(session_id);

        let formatted: Vec<String> = agents
            .iter()
            .map(|(name, desc)| {
                let marker = if *name == my_name { " (you)" } else { "" };
                format!("- {}{}: {}", name, marker, desc)
            })
            .collect();

        Ok(vec![Content::text(formatted.join("\n"))])
    }

    fn get_tools() -> Vec<Tool> {
        let send_schema = schema_for!(ChatSendParams);
        let send_schema_value =
            serde_json::to_value(send_schema).expect("Failed to serialize ChatSendParams schema");

        let read_schema = schema_for!(ChatReadParams);
        let read_schema_value =
            serde_json::to_value(read_schema).expect("Failed to serialize ChatReadParams schema");

        let claim_schema = schema_for!(ChatClaimParams);
        let claim_schema_value =
            serde_json::to_value(claim_schema).expect("Failed to serialize ChatClaimParams schema");

        let release_schema = schema_for!(ChatReleaseParams);
        let release_schema_value = serde_json::to_value(release_schema)
            .expect("Failed to serialize ChatReleaseParams schema");

        let list_schema = schema_for!(ChatListParams);
        let list_schema_value =
            serde_json::to_value(list_schema).expect("Failed to serialize ChatListParams schema");

        vec![
            Tool::new(
                "chat_send".to_string(),
                "Broadcast a message to all other subagents. Use this to declare what files \
                 you intend to modify or what actions you're taking, so others can avoid conflicts."
                    .to_string(),
                send_schema_value.as_object().unwrap().clone(),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("Send chat message".to_string()),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
            )),
            Tool::new(
                "chat_read".to_string(),
                "Read messages from other subagents. By default returns only new messages \
                 since your last read. Pass all=true to get the full transcript."
                    .to_string(),
                read_schema_value.as_object().unwrap().clone(),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("Read chat messages".to_string()),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
            )),
            Tool::new(
                "chat_claim".to_string(),
                "Atomically claim a resource (file, number, task). Returns success if you got it, \
                 or ERRORS if someone else already claimed it — you MUST pick a different resource. \
                 Use a simple canonical name (e.g. '3' not 'number 3', 'auth.py' not 'the auth module')."
                    .to_string(),
                claim_schema_value.as_object().unwrap().clone(),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("Claim resource".to_string()),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
            )),
            Tool::new(
                "chat_release".to_string(),
                "Release a previously claimed resource so other subagents can claim it."
                    .to_string(),
                release_schema_value.as_object().unwrap().clone(),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("Release claim".to_string()),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
            )),
            Tool::new(
                "chat_list".to_string(),
                "List all active subagents in the current session.".to_string(),
                list_schema_value.as_object().unwrap().clone(),
            )
            .annotate(ToolAnnotations::from_raw(
                Some("List chat agents".to_string()),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
            )),
        ]
    }
}

#[async_trait]
impl McpClientTrait for ChatClient {
    async fn list_tools(
        &self,
        _session_id: &str,
        _next_cursor: Option<String>,
        _cancellation_token: CancellationToken,
    ) -> Result<ListToolsResult, Error> {
        Ok(ListToolsResult {
            tools: Self::get_tools(),
            next_cursor: None,
            meta: None,
        })
    }

    async fn call_tool(
        &self,
        session_id: &str,
        name: &str,
        arguments: Option<JsonObject>,
        _working_dir: Option<&str>,
        _cancellation_token: CancellationToken,
    ) -> Result<CallToolResult, Error> {
        let content = match name {
            "chat_send" => self.handle_send(session_id, arguments).await,
            "chat_read" => self.handle_read(session_id, arguments).await,
            "chat_claim" => self.handle_claim(session_id, arguments).await,
            "chat_release" => self.handle_release(session_id, arguments).await,
            "chat_list" => self.handle_list(session_id).await,
            _ => Err(format!("Unknown tool: {}", name)),
        };

        match content {
            Ok(content) => Ok(CallToolResult::success(content)),
            Err(error) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Error: {}",
                error
            ))])),
        }
    }

    fn get_info(&self) -> Option<&InitializeResult> {
        Some(&self.info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ChatBus unit tests ────────────────────────────────────────────

    fn make_bus() -> ChatBus {
        ChatBus::new()
    }

    #[tokio::test]
    async fn test_send_and_read_messages() {
        let bus = make_bus();
        bus.register_agent("alice".into(), "desc".into()).await;
        bus.register_agent("bob".into(), "desc".into()).await;

        let msg = bus.send_message("alice".into(), "all".into(), "hello".into()).await;
        assert_eq!(msg.id, 1);
        assert_eq!(msg.from, "alice");

        // bob reads broadcast
        let msgs = bus.read_messages("bob", 0).await;
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "hello");

        // reading again with same cursor returns same messages
        let msgs = bus.read_messages("bob", 0).await;
        assert_eq!(msgs.len(), 1);

        // reading with updated cursor returns nothing
        let msgs = bus.read_messages("bob", 1).await;
        assert!(msgs.is_empty());
    }

    #[tokio::test]
    async fn test_read_filters_by_recipient() {
        let bus = make_bus();
        bus.send_message("alice".into(), "bob".into(), "for bob".into()).await;
        bus.send_message("alice".into(), "all".into(), "broadcast".into()).await;

        // bob sees both (addressed + broadcast)
        let msgs = bus.read_messages("bob", 0).await;
        assert_eq!(msgs.len(), 2);

        // charlie only sees broadcast
        let msgs = bus.read_messages("charlie", 0).await;
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "broadcast");
    }

    #[tokio::test]
    async fn test_message_ids_increment() {
        let bus = make_bus();
        let m1 = bus.send_message("a".into(), "all".into(), "1".into()).await;
        let m2 = bus.send_message("a".into(), "all".into(), "2".into()).await;
        let m3 = bus.send_message("a".into(), "all".into(), "3".into()).await;
        assert_eq!(m1.id, 1);
        assert_eq!(m2.id, 2);
        assert_eq!(m3.id, 3);
    }

    #[tokio::test]
    async fn test_read_since_id() {
        let bus = make_bus();
        bus.send_message("a".into(), "all".into(), "old".into()).await;
        bus.send_message("a".into(), "all".into(), "new".into()).await;

        let msgs = bus.read_messages("b", 1).await;
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "new");
    }

    // ── Claim tests ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_claim_success() {
        let bus = make_bus();
        let result = bus.claim("file.rs".into(), "alice".into()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_claim_denied() {
        let bus = make_bus();
        bus.claim("file.rs".into(), "alice".into()).await.unwrap();
        let result = bus.claim("file.rs".into(), "bob".into()).await;
        assert_eq!(result.unwrap_err(), "alice");
    }

    #[tokio::test]
    async fn test_claim_idempotent_same_owner() {
        let bus = make_bus();
        bus.claim("file.rs".into(), "alice".into()).await.unwrap();
        // Same owner trying again gets denied (not idempotent — it's claimed)
        let result = bus.claim("file.rs".into(), "alice".into()).await;
        assert_eq!(result.unwrap_err(), "alice");
    }

    #[tokio::test]
    async fn test_different_resources_independent() {
        let bus = make_bus();
        bus.claim("file1.rs".into(), "alice".into()).await.unwrap();
        let result = bus.claim("file2.rs".into(), "bob".into()).await;
        assert!(result.is_ok());
    }

    // ── Release tests ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_release_success() {
        let bus = make_bus();
        bus.claim("file.rs".into(), "alice".into()).await.unwrap();
        let result = bus.release("file.rs", "alice").await;
        assert!(result.is_ok());

        // Now bob can claim it
        let result = bus.claim("file.rs".into(), "bob".into()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_release_wrong_owner() {
        let bus = make_bus();
        bus.claim("file.rs".into(), "alice".into()).await.unwrap();
        let result = bus.release("file.rs", "bob").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("alice"));
    }

    #[tokio::test]
    async fn test_release_unclaimed() {
        let bus = make_bus();
        let result = bus.release("file.rs", "alice").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not claimed"));
    }

    // ── Agent registration tests ─────────────────────────────────────

    #[tokio::test]
    async fn test_register_and_list_agents() {
        let bus = make_bus();
        bus.register_agent("alice".into(), "architect".into()).await;
        bus.register_agent("bob".into(), "developer".into()).await;

        let agents = bus.list_agents().await;
        assert_eq!(agents.len(), 2);
    }

    #[tokio::test]
    async fn test_register_idempotent() {
        let bus = make_bus();
        bus.register_agent("alice".into(), "first".into()).await;
        bus.register_agent("alice".into(), "second".into()).await;

        let agents = bus.list_agents().await;
        assert_eq!(agents.len(), 1);
        // Keeps the first description
        assert_eq!(agents[0].1, "first");
    }

    // ── Agent name hashing ───────────────────────────────────────────

    fn hash_agent_name(session_id: &str) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        session_id.hash(&mut hasher);
        let hash = hasher.finish();
        format!("agent-{:06x}", hash & 0xFFFFFF)
    }

    #[test]
    fn test_agent_name_format() {
        let name = hash_agent_name("20260306_42");
        assert!(name.starts_with("agent-"));
        // "agent-" (6) + 6 hex chars = 12
        assert_eq!(name.len(), 12);
    }

    #[test]
    fn test_agent_name_deterministic() {
        let name1 = hash_agent_name("session-123");
        let name2 = hash_agent_name("session-123");
        assert_eq!(name1, name2);
    }

    #[test]
    fn test_agent_name_unique() {
        let name1 = hash_agent_name("session-1");
        let name2 = hash_agent_name("session-2");
        assert_ne!(name1, name2);
    }

    // ── Tool listing ─────────────────────────────────────────────────

    #[test]
    fn test_get_tools_count() {
        let tools = ChatClient::get_tools();
        assert_eq!(tools.len(), 5); // send, read, claim, release, list
    }

    #[test]
    fn test_get_tools_names() {
        let tools = ChatClient::get_tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
        assert!(names.contains(&"chat_send"));
        assert!(names.contains(&"chat_read"));
        assert!(names.contains(&"chat_claim"));
        assert!(names.contains(&"chat_release"));
        assert!(names.contains(&"chat_list"));
    }
}
