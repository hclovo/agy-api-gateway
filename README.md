# agy-api-gateway

基于 Google Antigravity 官方 CLI（`agy`）的本地/自托管多协议 API 网关。  
可将你本机已登录的 Antigravity 账号能力，通过标准 `base_url` + 独立签发的 `api_key` 分发给其他 Agent Harness、IDE 插件、OpenAI / Gemini / Anthropic SDK 或脚本使用。

## 特性

- **三协议兼容（含多模态图片 / 视频 / PDF / 音频读取）**：
  - **OpenAI 兼容**：`GET /v1/models`、`POST /v1/chat/completions`（支持非流式 + SSE `stream: true` + `tools` + `image_url` / `video_url` / `input_audio` / `file`）
  - **Gemini 原生 REST 兼容**：`GET /v1beta/models`、`POST /v1beta/models/{model}:generateContent`、`:streamGenerateContent?alt=sse`（支持 `inlineData` / `fileData` 图片与视频）
  - **Anthropic Messages 兼容**：`POST /v1/messages`（支持流式与非流式 + `tools` + `image` / `document` / `video` 及 `tool_result` 内嵌截图）
- **多账号池管理与容灾路由**：
  - 每个 Antigravity 账号使用独立隔离的 `HOME` 目录（`./data/accounts/<id>`），默认保留既有主账号（`default`）无缝兼容
  - 支持 `least_busy`（最少并发优先）、`round_robin`（均匀轮询）、`priority`（优先级）三种路由策略，遇配额/429/鉴权失败时首字前自动冷却并无缝故障转移到健康账号
  - 内置 Web 控制台（`/ui`）：支持在浏览器直接拉起官方 `agy acp --login` 获取 Google 授权链接并回填授权码完成新账号登录、一键健康检测、解冷、编辑及 Key 账号绑定
- **API Key 分发与控制（支持 PostgreSQL / SQLite 双后端）**：
  - 支持 PostgreSQL（通过 `AGY_GATEWAY_DATABASE_URL` 配置连接池）或本地 SQLite，内置 `agy-gateway db migrate` 一键无损迁移工具
  - 数据库仅保存 API Key 的 SHA-256 哈希，支持创建 / 编辑 / 禁用 / 删除 Key
  - 支持单 Key 每分钟请求限流（RPM）、模型白名单、指定账号白名单、请求与 Token 用量统计
- **并发调度、长会话智能上下文管理与磁盘保洁**：
  - 默认 `12` 并发、`300s` 排队超时、`600s` 单请求超时、SSE `10s` 心跳保活、非流式断连自动回收子进程
  - 融合 **Pi (`pi-mono` / `pi-vcc`)**、**DeepSeek Harness** 与 **Paseo** 的长会话上下文工程，并支持 **`agy --conversation` 原生多轮会话复用**：
    - **原生会话复用与增量转发（默认开启）**：在 PostgreSQL / SQLite（`gateway_sessions` 表）维护会话游标，同一会话后续轮次自动复用原 `agy` 会话 ID 并仅发送本轮增量 Delta，彻底突破单条 `191.5 KB` 限制、用满 1M 原生上下文窗口与 Prompt Cache；遇到历史编辑/回滚、模型切换或会话失效时自动无感回退全量上下文模式，后台保洁自动保护活跃会话 `.db`（默认 24 小时）
    - **三协议结构化工具解析**：OpenAI / Anthropic / Gemini 均保留结构化 `tool_call` / `tool_result` 元数据与会话指纹，并采用 JSON-aware 扫描器准确提取含转义或代码块的 `<tool_call>`
    - **工具输出去重与观察掩码（Observation Masking）**：自动合并同一文件的重复历史读取，对过长历史工具输出保留路径、退出状态、错误堆栈与首尾摘要，避免单次大输出冲垮整段对话
    - **原子轮次切分（`findCutPoint` + `isSplitTurn`）**：严格禁止切断 `assistant(tool_call)` 与 `tool_result` 的配对关系，并在长工具链中途切分时自动保留首轮目标（Goal）与当前轮次原始用户指令
    - **结构化检查点与粘性前缀缓存**：将被压缩的历史沉淀为结构化检查点（目标与指令、已读/已改文件、已执行命令、关键决策、未解决报错），支持 `deterministic`（零额外额度/零死锁，默认）、`hybrid`（后台异步 LLM 升级 + SQLite 缓存）与 `llm`（分块 Map/Reduce 防死锁摘要）三种模式，并跨轮复用粘性前缀以最大化 Prompt Cache 命中
    - **上下文溢出自愈与协议原生错误返回**：首字前命中上游上下文溢出时不误伤账号冷却，自动收紧预算重试（默认最多 2 次）；仍无法容纳时返回协议原生的 HTTP `400`（OpenAI `context_length_exceeded` / Anthropic `prompt is too long` / Gemini `INVALID_ARGUMENT`）以正确触发客户端 `/compact`；同时修复流式中断与截断 `<tool_call>` 时的 `finish_reason`（`length` / `max_tokens` / `MAX_TOKENS`）及 Anthropic SSE `input_tokens` 上报
  - 后台定时自动清理各账号下过期的无状态会话 `.db`、日志与过期上下文检查点（默认保留最近 30 分钟），防止磁盘持续膨胀与启动变慢

## 快速开始

```bash
cd /path/to/agy-api-gateway
uv sync

# 1. 查看可用模型与内置别名
uv run agy-gateway models

# 2. 签发一个 API Key（返回 base_url 和 sk-agy-... 密钥）
uv run agy-gateway key create --name my-harness --rpm 30

# 3. 启动网关服务
uv run agy-gateway serve --host 127.0.0.1 --port 8787
```

## 其他 Harness 接入方式

假设创建 Key 时获得 `API_KEY=sk-agy-xxxx`：

### 1. OpenAI 兼容客户端（OpenCode / Cline / Roo / Python `openai` SDK / LiteLLM 等）
- `base_url`: `http://127.0.0.1:8787/v1`
- `api_key`: `sk-agy-xxxx`
- `model`: `gemini-3.8-flash-high`、`gemini-4-hydrogen-high` 或别名 `gemini-2.5-flash` / `gemini-2.5-pro`

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="sk-agy-xxxx")
resp = client.chat.completions.create(
    model="gemini-3.8-flash-high",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

### 2. Gemini 原生 REST 客户端
- `base_url`: `http://127.0.0.1:8787/v1beta`
- 鉴权支持 `x-goog-api-key: sk-agy-xxxx`、`Authorization: Bearer sk-agy-xxxx` 或 `?key=sk-agy-xxxx`

### 3. Anthropic 兼容客户端
- `ANTHROPIC_BASE_URL=http://127.0.0.1:8787`
- `ANTHROPIC_API_KEY=sk-agy-xxxx`

## 如何对外提供公网 `base_url`

既可以直接通过项目内置的 `docker-compose.yml` + `Caddyfile` 在云服务器上部署（自动申请 HTTPS 证书），也可以在本地运行 `agy-gateway`，通过 SSH 反向隧道把端口映射到公网服务器（再由 Nginx/Caddy 反代加 HTTPS）：

```bash
# 本机启动网关
uv run agy-gateway serve --host 127.0.0.1 --port 8787

# 将远端 127.0.0.1:18787 转发到本机 127.0.0.1:8787
ssh -N -R 127.0.0.1:18787:127.0.0.1:8787 your-vps
```

或在公网服务器上配置 Nginx 反向代理到 `http://127.0.0.1:18787`（注意关闭缓冲：`proxy_buffering off;` 以保证 SSE 流式响应实时输出）。

## 常用 Key 管理命令

```bash
uv run agy-gateway key list
uv run agy-gateway key disable <key_id>
uv run agy-gateway key enable <key_id>
uv run agy-gateway key delete <key_id>
```

## 运行测试

```bash
uv run pytest
```

## 变更记录

详见 [CHANGELOG.md](./CHANGELOG.md)。
