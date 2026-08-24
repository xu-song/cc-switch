//! 代理完整请求/响应日志记录
//!
//! 把客户端发往代理的请求体与上游的响应体（流式聚合后/非流式原样）一起落到
//! 本地 JSONL（`~/.cc-switch/proxy_full_logs/<app_type>/<session-id>.<perspective>.jsonl`），
//! 一个会话每个视点一个文件，方便事后排查 prompt / tool_use / reasoning 的完整内容。
//!
//! - 文件名用 session-id + 视点后缀（`upstream` / `client`）。两视点同时开启时
//!   各落一个文件，分别对应转换前/后的报文，互不混写。session-id 与各 CLI 自己写的
//!   `<session-id>.jsonl` 日志对应，Codex 去掉内部 `codex_` 前缀，session 缺失落
//!   `unknown-session.<perspective>.jsonl`。
//! - 与 `proxy_request_logs` 表的元数据互补：那张表只存 token/cost/latency，
//!   这里存完整 body。共用同一个 `request_id` 字段（UUIDv4）做关联。
//! - 写入用每文件路径一把短时锁；失败仅 warn，不影响转发。
//! - 默认关闭，由前端开关写入 `proxy_config.full_logging_enabled` 列。

pub mod aggregators;

use chrono::Utc;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex as StdMutex,
};
use tokio::sync::Mutex as AsyncMutex;

pub use aggregators::{aggregator_for_api_format, select_aggregator, SseAggregator};

/// 流式聚合的累计输入字节软上限。与非流式路径的 `MAX_RESPONSE_BODY_BYTES` 对齐：
/// 超过则停止聚合并标记 truncated，流照常透传，仅 full-log 不再记录后续内容。
/// 聚合状态会完整驻留内存，故用同一上限防止超长/恶意流式响应导致 OOM。
const MAX_AGGREGATED_BYTES: usize = 128 * 1024 * 1024;

/// 一条完整日志记录
#[derive(Debug, Clone, Serialize)]
pub struct FullLogRecord {
    pub ts: String,
    #[serde(rename = "requestId")]
    pub request_id: String,
    #[serde(rename = "sessionId")]
    pub session_id: String,
    #[serde(rename = "providerId")]
    pub provider_id: String,
    #[serde(rename = "appType")]
    pub app_type: String,
    /// 记录视点：`"upstream"`（转换后请求 + 上游原始响应）或
    /// `"client"`（客户端原始请求 + 转换后响应）。两视点同开时，一次请求会落两条
    /// 记录，靠本字段区分。
    pub perspective: String,
    pub endpoint: String,
    pub model: String,
    #[serde(rename = "durationMs")]
    pub duration_ms: u64,
    #[serde(rename = "isStreaming")]
    pub is_streaming: bool,
    pub request: Value,
    pub response: Value,
    #[serde(rename = "statusCode")]
    pub status_code: u16,
    pub error: Option<String>,
    /// session_id 是否由客户端真实提供。兜底生成的 UUID 不能用来命名会话文件
    /// （它和 CLI 写的日志文件名对不上），这类记录落到 `<app>/unknown-session.jsonl`。
    #[serde(skip)]
    pub session_client_provided: bool,
}

impl FullLogRecord {
    /// 构建一条记录。`ts` 自动填当前 UTC。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_id: String,
        session_id: String,
        session_client_provided: bool,
        provider_id: String,
        app_type: String,
        perspective: super::FullLogPerspective,
        endpoint: String,
        model: String,
        duration_ms: u64,
        is_streaming: bool,
        request: Value,
        response: Value,
        status_code: u16,
        error: Option<String>,
    ) -> Self {
        Self {
            ts: Utc::now().to_rfc3339(),
            request_id,
            session_id,
            session_client_provided,
            provider_id,
            app_type,
            perspective: perspective.as_str().to_string(),
            endpoint,
            model,
            duration_ms,
            is_streaming,
            request,
            response,
            status_code,
            error,
        }
    }
}

/// 全局每文件路径锁注册表。避免同一天文件被多个写入并发追加导致行被打散。
static FILE_LOCKS: once_cell::sync::Lazy<StdMutex<HashMap<PathBuf, Arc<AsyncMutex<()>>>>> =
    once_cell::sync::Lazy::new(|| StdMutex::new(HashMap::new()));

fn lock_for(path: &Path) -> Arc<AsyncMutex<()>> {
    let mut guard = FILE_LOCKS.lock().expect("file locks registry poisoned");
    guard
        .entry(path.to_path_buf())
        .or_insert_with(|| Arc::new(AsyncMutex::new(())))
        .clone()
}

/// 把记录追加到对应会话文件。失败仅 warn。
///
/// 路径布局：`<full_log_dir>/<app_type>/<session-id>.jsonl`，一个会话一个文件，
/// 与各 CLI 自己写的 `<session-id>.jsonl` 日志文件名对应，方便对照查看。
/// session_id 缺失（兜底生成的 UUID）时落到 `<app_type>/unknown-session.jsonl`。
///
/// 该函数在 tokio 异步上下文里调用，但落盘用 `spawn_blocking` 移到阻塞线程池，
/// 避免阻塞 reactor 线程。
pub async fn append_record(record: FullLogRecord) {
    let dir = match crate::config::get_proxy_full_log_dir() {
        Ok(d) => d,
        Err(e) => {
            log::warn!("[full_logger] 获取日志目录失败: {e}");
            return;
        }
    };

    let app_dir = dir.join(sanitize_path_component(&record.app_type));
    let file_stem = session_file_stem(&record);
    // 按视点分文件：`<session-id>.<perspective>.jsonl`。
    // 两视点同时开启时各落一个文件，分别对应转换前/后的报文，互不混写。
    let path = app_dir.join(format!("{file_stem}.{}.jsonl", record.perspective));

    let mut line = match serde_json::to_string(&record) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("[full_logger] 序列化记录失败: {e}");
            return;
        }
    };
    line.push('\n');

    let path_for_blocking = path.clone();
    let lock = lock_for(&path);
    let _held = lock.lock().await;
    let result = tokio::task::spawn_blocking(move || write_line(&path_for_blocking, &line)).await;
    match result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => log::warn!("[full_logger] 写入失败 {}: {e}", path.display()),
        Err(e) => log::warn!("[full_logger] 写入任务 join 失败: {e}"),
    }
}

/// 计算会话文件名（不含扩展名）。
///
/// - 客户端未提供稳定 session_id（兜底 UUID）→ `unknown-session`
/// - 去掉 Codex 内部加的 `codex_` 前缀，让文件名与 Codex CLI 的 `<uuid>.jsonl` 对应
/// - 做文件名安全清洗（防止 `/`、`..` 等穿越目录）
fn session_file_stem(record: &FullLogRecord) -> String {
    if !record.session_client_provided {
        return "unknown-session".to_string();
    }
    let raw = record
        .session_id
        .strip_prefix("codex_")
        .unwrap_or(&record.session_id);
    let cleaned = sanitize_path_component(raw);
    if cleaned.is_empty() {
        "unknown-session".to_string()
    } else {
        cleaned
    }
}

/// 文件名/目录名安全清洗：只保留字母数字与 `-` `_` `.`，其余替换为 `_`。
/// 同时拒绝纯 `.` / `..`，避免目录穿越。
fn sanitize_path_component(input: &str) -> String {
    let mut out: String = input
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    // 截断过长文件名（部分文件系统 255 字节上限），留出扩展名空间
    if out.len() > 200 {
        out.truncate(200);
    }
    if out.is_empty() || out == "." || out == ".." || out.chars().all(|c| c == '.') {
        return String::new();
    }
    out
}

fn write_line(path: &Path, line: &str) -> std::io::Result<()> {
    use std::io::Write;
    // 确保 per-app 子目录存在（顶层目录已由 get_proxy_full_log_dir 创建）
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut opts = std::fs::OpenOptions::new();
    opts.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut file = opts.open(path)?;
    file.write_all(line.as_bytes())?;
    file.flush()?;
    Ok(())
}

// ============================================================================
// SSE 流式聚合 collector + finish guard（与 SseUsageCollector 同款形态）
// ============================================================================

type AggregatorSlot = Arc<AsyncMutex<Option<Box<dyn SseAggregator + Send>>>>;

#[derive(Clone)]
pub struct SseFullLogCollector {
    inner: Arc<SseFullLogCollectorInner>,
}

struct SseFullLogCollectorInner {
    aggregator: AggregatorSlot,
    finished: AtomicBool,
    /// 累计喂入的 SSE event 字节数，超 `MAX_AGGREGATED_BYTES` 后停止 ingest。
    accumulated_bytes: AtomicUsize,
    /// 是否因超限停止了聚合。finalize 时据此在产物里标记 truncated。
    truncated: AtomicBool,
    on_complete: Box<dyn Fn(Value) + Send + Sync + 'static>,
}

impl SseFullLogCollector {
    /// 创建一个流式 full-log 收集器。
    ///
    /// `aggregator` 由 `select_aggregator(app_type, endpoint)` 提供。
    /// `on_complete` 在流结束（或被 Drop guard 触发）时拿到聚合后的完整 response JSON。
    pub fn new(
        aggregator: Box<dyn SseAggregator + Send>,
        on_complete: impl Fn(Value) + Send + Sync + 'static,
    ) -> Self {
        Self {
            inner: Arc::new(SseFullLogCollectorInner {
                aggregator: Arc::new(AsyncMutex::new(Some(aggregator))),
                finished: AtomicBool::new(false),
                accumulated_bytes: AtomicUsize::new(0),
                truncated: AtomicBool::new(false),
                on_complete: Box::new(on_complete),
            }),
        }
    }

    /// 推入一个解析过的 SSE event JSON。
    pub async fn push(&self, value: Value) {
        let mut guard = self.inner.aggregator.lock().await;
        let Some(agg) = guard.as_mut() else {
            return;
        };
        // 超限后停止 ingest：流照常透传，仅不再记录后续 event。
        if self.inner.truncated.load(Ordering::Relaxed) {
            return;
        }
        let chunk_bytes = value.to_string().len();
        let prev = self.inner.accumulated_bytes.fetch_add(chunk_bytes, Ordering::Relaxed);
        if prev + chunk_bytes > MAX_AGGREGATED_BYTES {
            self.inner.truncated.store(true, Ordering::Relaxed);
            return;
        }
        agg.ingest_event(&value);
    }

    /// 结束聚合并触发回调。幂等。
    pub async fn finish(&self) {
        if self.inner.finished.swap(true, Ordering::SeqCst) {
            return;
        }
        let aggregator = {
            let mut guard = self.inner.aggregator.lock().await;
            guard.take()
        };
        if let Some(agg) = aggregator {
            let mut value = agg.finalize();
            if self.inner.truncated.load(Ordering::Relaxed) {
                if let Some(obj) = value.as_object_mut() {
                    obj.insert(
                        "fullLogTruncated".to_string(),
                        Value::Bool(true),
                    );
                }
            }
            (self.inner.on_complete)(value);
        }
    }
}

/// RAII guard，确保 stream 在中途被丢弃（客户端断开等）时也会触发 finish。
/// 与 `SseUsageFinishGuard` 同款形态。
pub struct SseFullLogFinishGuard {
    collector: Option<SseFullLogCollector>,
}

impl SseFullLogFinishGuard {
    pub fn new(collector: SseFullLogCollector) -> Self {
        Self {
            collector: Some(collector),
        }
    }

    pub fn disarm(&mut self) {
        self.collector = None;
    }
}

impl Drop for SseFullLogFinishGuard {
    fn drop(&mut self) {
        if let Some(collector) = self.collector.take() {
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn(async move {
                    collector.finish().await;
                });
            } else {
                log::warn!("Full-log 收尾保护触发时 Tokio runtime 不可用，跳过异步 finish");
            }
        }
    }
}

/// 旁路（tee）一条上游原始 SSE 字节流：原样透传每个 chunk，同时把解析出的
/// SSE event JSON 喂给 `collector`，流结束时触发其 finalize 落盘。
///
/// 用于 Claude transform 路径——上游是 OpenAI/Responses SSE，会被转换器改写成
/// Anthropic SSE 后再返回客户端。full-log 需要的是**转换前**的上游原始报文，
/// 所以在转换器消费之前先在这里旁路一份。
///
/// 与 `response_processor::create_logged_passthrough_stream` 的区别：那个挂在
/// 转换**后**的流上（记 Anthropic 形态、走 usage 收集器）；这个挂在转换**前**
/// 的上游流上（记上游原始形态）。两者互补，互不干扰。
pub fn tee_raw_sse_for_full_log<S>(
    stream: S,
    collector: SseFullLogCollector,
) -> impl futures::Stream<Item = Result<bytes::Bytes, std::io::Error>> + Send
where
    S: futures::Stream<Item = Result<bytes::Bytes, std::io::Error>> + Send + 'static,
{
    use crate::proxy::sse::{append_utf8_safe, strip_sse_field, take_sse_block};
    use futures::StreamExt;

    async_stream::stream! {
        let mut buffer = String::new();
        let mut utf8_remainder: Vec<u8> = Vec::new();
        let mut finish_guard = SseFullLogFinishGuard::new(collector.clone());

        tokio::pin!(stream);

        while let Some(item) = stream.next().await {
            match item {
                Ok(bytes) => {
                    append_utf8_safe(&mut buffer, &mut utf8_remainder, &bytes);
                    while let Some(block) = take_sse_block(&mut buffer) {
                        if block.trim().is_empty() {
                            continue;
                        }
                        for line in block.lines() {
                            if let Some(data) = strip_sse_field(line, "data") {
                                let data = data.trim();
                                if data.is_empty() || data == "[DONE]" {
                                    continue;
                                }
                                if let Ok(value) = serde_json::from_str::<Value>(data) {
                                    collector.push(value).await;
                                }
                            }
                        }
                    }
                    yield Ok(bytes);
                }
                Err(e) => {
                    yield Err(e);
                }
            }
        }

        collector.finish().await;
        finish_guard.disarm();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn record_with(session_id: &str, client_provided: bool, app_type: &str) -> FullLogRecord {
        FullLogRecord::new(
            "req-1".to_string(),
            session_id.to_string(),
            client_provided,
            "provider-1".to_string(),
            app_type.to_string(),
            super::super::FullLogPerspective::Upstream,
            "/v1/messages".to_string(),
            "claude-opus-4-7".to_string(),
            0,
            true,
            json!({}),
            json!({}),
            200,
            None,
        )
    }

    #[test]
    fn session_file_stem_uses_session_id_when_client_provided() {
        let r = record_with("abc-123-def", true, "claude");
        assert_eq!(session_file_stem(&r), "abc-123-def");
    }

    #[test]
    fn session_file_stem_strips_codex_prefix() {
        let r = record_with("codex_9f8e7d6c-uuid", true, "codex");
        assert_eq!(session_file_stem(&r), "9f8e7d6c-uuid");
    }

    #[test]
    fn session_file_stem_falls_back_when_not_client_provided() {
        // 兜底生成的 UUID（client_provided=false）不能当会话名
        let r = record_with("random-generated-uuid", false, "claude");
        assert_eq!(session_file_stem(&r), "unknown-session");
    }

    #[test]
    fn session_file_stem_falls_back_when_empty_after_clean() {
        let r = record_with("///", true, "claude");
        // 清洗后是 "___"，非空，所以保留清洗结果而非兜底
        assert_eq!(session_file_stem(&r), "___");
    }

    #[test]
    fn sanitize_rejects_path_traversal() {
        assert_eq!(sanitize_path_component(".."), "");
        assert_eq!(sanitize_path_component("."), "");
        assert_eq!(sanitize_path_component("a/../b"), "a_.._b");
        assert_eq!(sanitize_path_component("sess/../../etc"), "sess_.._.._etc");
    }

    #[test]
    fn sanitize_keeps_safe_chars() {
        assert_eq!(
            sanitize_path_component("01HX9-abc_DEF.123"),
            "01HX9-abc_DEF.123"
        );
    }

    #[test]
    fn session_id_is_serialized_in_record() {
        let r = record_with("sess-42", true, "claude");
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["sessionId"], "sess-42");
        // session_client_provided 标记不应出现在落盘 JSON 里
        assert!(v.get("session_client_provided").is_none());
        assert!(v.get("sessionClientProvided").is_none());
    }

    #[test]
    fn perspective_is_serialized_as_lowercase_string() {
        let upstream = record_with("sess-1", true, "claude");
        assert_eq!(
            serde_json::to_value(&upstream).unwrap()["perspective"],
            "upstream"
        );

        let mut client = record_with("sess-1", true, "claude");
        client.perspective = super::super::FullLogPerspective::Client.as_str().to_string();
        assert_eq!(
            serde_json::to_value(&client).unwrap()["perspective"],
            "client"
        );
    }
}
