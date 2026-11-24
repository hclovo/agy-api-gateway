# Changelog

本文档记录 `agy-api-gateway` 的主要功能演进、架构升级与问题修复，格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 规范。

---

## [0.4.0] - 2026-11-24

### Added
- **原生多轮会话复用（`agy --conversation` 增量转发）**
  - 新增 `agy_gateway/sessions.py` 与 `gateway_sessions` 数据表，按 `(api_key_id, session_fingerprint, model)` 严格隔离记录会话游标（账号归属、`agy_conversation_id`、已同步消息数与前缀哈希、助手回复指纹）。
  - 同一会话后续轮次自动路由回原账号并通过 `agy --conversation <id>` 仅发送本轮新增 Delta 消息（仅暂存 Delta 引用的新媒体附件），突破单条 `USER_INPUT` `191,500` 字节上限，原生用满 **1M 上下文窗口**、`agy` 原生自动压缩与 Prompt Cache。
  - 新增双重防失联守卫：恢复前校验磁盘 `<HOME>/.gemini/antigravity-cli/conversations/<cid>.db` 存在性，运行期在 `AgyRunner` 校验 `init` 事件会话 ID（不一致时触发 `AgySessionMismatchError`），首字前自动无感回退至全量上下文模式且不冷却账号。
  - 新增配置项 `AGY_GATEWAY_SESSION_REUSE_ENABLED`（默认 `true`）与 `AGY_GATEWAY_SESSION_TTL_HOURS`（默认 `24`），响应头新增 `X-Agy-Session-Reused`（`true`/`false`）与 `X-Agy-Context-Stage: session_delta`。
  - `housekeeping.py` 支持传入 `protected_conversation_ids`，定时清理时自动跳过活跃复用会话的 `conversations/`、`brain/`、`presence/` 文件，避免长会话中途被 30 分钟默认保洁误删。
- **PostgreSQL / SQLite 双后端存储与一键迁移**
  - 新增 `agy_gateway/db.py`（`DatabaseBackend`），配置 `AGY_GATEWAY_DATABASE_URL=postgresql://...` 时自动启用 `psycopg[binary,pool]` 连接池（含探活 `check_connection`、TCP keepalive 与 TLS 加密），未配置时自动回退本地 SQLite。
  - `api_keys`、`accounts`、`gateway_settings`、`context_checkpoints`、`gateway_sessions` 全面适配 PostgreSQL（计数器升级 `BIGINT`、浮点时间戳升级 `DOUBLE PRECISION`、LRU 淘汰改为标准多列 `IN` 语法）。
  - 新增幂等数据迁移命令 `agy-gateway db migrate --sqlite ... --target-url ...`，并在 `/health` 与 `/admin/stats` 暴露 `database_backend`。
- **测试与控制台增强**
  - 新增 `tests/test_db.py` 与 `tests/test_session_reuse.py`；Web 控制台「长会话上下文保护」卡片升级展示原生复用命中次数、活跃会话数与兜底压缩统计。

### Fixed
- **修复 `<system-reminder>` / `SOUL.md` 模板挤掉首轮用户真实需求的 Bug**：新增 `extract_clean_human_intent()`，在生成检查点与会话指纹前剥离嵌套的 `<system-reminder>`、`<identity_context>`、`<user_info>`、`<environment_details>` 及内联行号代码倾倒，避免客户端注入的模板噪声遮盖真实用户指令或引发会话指纹碰撞。
- **修复含图片会话 `prefix_hash` 轮轮变化的 Bug**：计算 `prefix_hash` 前归一化临时媒体暂存路径 `/media_<16hex>/`，恢复粘性前缀检查点与上游 Prompt Cache 命中率。
- **修复 Windows 路径解析被误截断的 Bug**：去除行号后缀时仅匹配末尾 `:\d*-\d*$`，保留 `D:\...` 盘符路径完整展示；过滤误入助手里程碑的原始工具包装文本。

---

## [0.3.0] - 2026-11-23

### Added
- **渐进式长会话上下文管理引擎（融合 Pi / DeepSeek Harness / Paseo 实践）**
  - 新增 `agy_gateway/context_manager.py`，在回退无状态拼装时按 Stage 0（无损直通）→ Stage 1（工具去重与分级压缩、清理陈旧 `<thinking>`）→ Stage 2（重复文件读取去重 `[Superseded ...]`、历史大工具输出观察掩码、单条超大最新 `tool_result` 预算保护）→ Stage 3（Pi `findCutPoint` + `isSplitTurn` 原子轮次切分，严禁切断 `assistant(tool_call)` ↔ `tool_result` 配对）→ Stage 4（结构化检查点 + `context_checkpoints` 粘性前缀缓存）逐级保护。
  - 支持三种压缩模式（`AGY_GATEWAY_COMPACTION_MODE`）：`deterministic`（零额外额度、毫秒级）、`hybrid`（借鉴 `pi-async-compaction`，首轮即时返回确定性检查点并后台异步 LLM 升级摘要）、`llm`（借鉴 DeepSeek Harness 分块有界 Map/Reduce，彻底消除摘要调用本身溢出的死锁，异常时自动降级）。
  - CJK 加权保守 Token 估算（修正中日韩文本低估问题）；历史已压缩轮次的媒体附件标记为归档，不再强制模型重复 `read_file`。
- **上下文溢出自愈与协议原生错误返回**
  - 上游命中 `CONTEXT_WINDOW_EXCEEDED` / `prompt is too long` 等溢出报错时，首字前**不计错、不冷却账号**，自动按 `0.6x` 逐步收紧预算重试（默认最多 `AGY_GATEWAY_OVERFLOW_MAX_RETRIES=2` 次）。
  - 重试仍无法容纳时，返回各协议原生 HTTP `400`（及对应 SSE 错误事件）：OpenAI `code: "context_length_exceeded"`、Anthropic `invalid_request_error` + `"prompt is too long: ..."`、Gemini `400 INVALID_ARGUMENT`，便于下游 Harness 自动触发 `/compact`。
- **可观测性与运维接口**
  - 新增 `X-Agy-Context-*` 诊断响应头、`GET /admin/context/stats`、`POST /admin/context/cache/clear`，并在 `/ui` 控制台新增「长会话上下文保护」实时统计卡片；新增 `tests/test_context_manager.py`（13 项长会话专项及 200 轮 >1.5 MB 压测）。

### Fixed
- **三协议结构化工具解析**：`parse_anthropic_request` 与 `parse_gemini_request` 保留结构化 `tool_calls` / `tool_results`（并清理重放的历史 `thinking`/`thought` 块），与 OpenAI 对齐。
- **JSON-aware `<tool_call>` 扫描器与截断检测**：改用 `JSONDecoder.raw_decode` 解析 `<tool_call>`，修复代码参数内含字面量 `</tool_call>` 或 Markdown 代码围栏时解析失败的问题；检测未闭合的截断 `<tool_call>` 并映射为 `length` / `max_tokens` / `MAX_TOKENS`。
- **修复 Anthropic SSE `input_tokens` 恒为 0 的问题**：在 `message_start` 与 `message_delta` 中上报真实 `input_tokens` 与 `cache_read_input_tokens`；OpenAI `has_tools=True` 流式模式实时透传 `reasoning_content`。
- **修复 `agy` 进程异常提前 EOF 被误判为正常结束的问题**：`agy_client.py` 行缓冲提升至 16 MiB，未收到 `result` 且无输出时抛错触发故障转移，已有部分输出时标注 `finish_reason="length"`。

---

## [0.2.1] - 2026-11-23

### Added
- **Caddy 自动 HTTPS 反向代理与域名部署**：新增 `Caddyfile` 并完善 `docker-compose.yml`（整合 `mihomo`、`agy-gateway`、`caddy` 三容器编排），支持 100 MB 请求体、关闭流式缓冲（`flush_interval -1`）与长连接零写超时。

---

## [0.2.0] - 2026-11-19

### Added
- **多账号池与容灾路由（`accounts.py`）**：支持为每个 Antigravity 账号分配独立隔离 `HOME` 目录，提供 `least_busy`、`round_robin`、`priority` 三种调度策略；遇 429 / 配额耗尽 / 503 / 鉴权失效时在首字前自动冷却并无缝故障转移。
- **浏览器交互式 OAuth 登录与额度监控**：内置 Web 管理控制台（`/ui`），支持在浏览器直接拉起 `agy acp --login` 完成新账号授权、批量健康检测、手动解冷、Key 绑定账号，以及基于只读 `agy /quota` 与 `/credits` 的 5 小时/周额度余量定时监控。
- **多模态媒体附件支持（`media.py`）**：三协议全面支持图片、视频、音频、PDF（含 Anthropic `tool_result` 内嵌截图）提取、SSRF 防护校验、按请求隔离暂存并交由 `read_file` 读取，请求结束后自动清理。
- **磁盘保洁与容器化部署（`housekeeping.py`、`Dockerfile`）**：后台定时清理过期的 `agy` 会话 `.db`、日志与孤儿媒体目录；提供分层缓存 `Dockerfile` 与 `docker-compose.yml`。

---

## [0.1.0] - 2026-11-18

### Added
- **多协议网关初始版本**：基于 Google Antigravity 官方 CLI（`agy` `stream-json` 模式），同时兼容 OpenAI（`/v1/models`、`/v1/chat/completions`）、Gemini（`/v1beta/models`、`:generateContent`、`:streamGenerateContent`）与 Anthropic（`/v1/messages`）流式与非流式协议及工具调用。
- **API Key 分发与流控（`keys.py`、`cli.py`）**：SQLite 仅保存 SHA-256 哈希，支持创建/编辑/禁用/删除 Key、滑动窗口 RPM 限流、模型白名单与请求/Token 用量统计。
- **并发调度与基础 Prompt 预算**：全局并发信号量、排队超时、SSE 心跳保活、非流式断连回收子进程，以及大工具 Schema 编译压缩。
