# EQBench3 本地 LLM API 适配与启动说明

## 1. 本次已完成的改动

1. 改造了 API URL 兼容逻辑（`utils/api.py`）  
- 现在 `TEST_API_URL` / `JUDGE_API_URL` 支持以下写法：  
  - `http://127.0.0.1:18080`  
  - `http://127.0.0.1:18080/v1`  
  - `http://127.0.0.1:18080/v1/chat/completions`  
- 代码会自动归一化到 `.../v1/chat/completions`，兼容 OpenAI 风格接口地址配置。

2. 增强了响应解析兼容（`utils/api.py`）  
- 兼容常见 OpenAI 风格响应（`choices[0].message.content`、分段 content、`output_text`、`output` 结构）。

3. 支持从 `.env` 读取模型名默认值（`eqbench3.py`）  
- `--test-model` 默认读取 `TEST_MODEL_NAME`  
- `--judge-model` 默认读取 `JUDGE_MODEL_NAME`  
- 如果没传 `--test-model` 且 `.env` 里也没有，会明确报错。

4. 新增轻量联通测试脚本（`scripts/smoke_test_api.py`）  
- 会分别测试 `test/judge` 两套配置。  
- 每套会做两种调用：  
  - 通过项目内部 `APIClient` 调用  
  - 通过 `requests.post` 直接调用  
- 只发极小请求，避免高额 token 消耗。

## 2. 已执行的环境与验证

1. 用 `uv` 构建环境并安装依赖：  
```powershell
uv venv .venv
uv pip install -r requirements.txt --python .venv\Scripts\python.exe
```

2. 运行了联通测试：  
```powershell
uv run --python .venv\Scripts\python.exe python scripts/smoke_test_api.py --type both --verbosity DEBUG
```

3. 联通测试结果（已通过）  
- test: `status = ok`，返回 `OK`  
- judge: `status = ok`，返回 `OK`  
- 归一化后实际请求地址：`http://127.0.0.1:18080/v1/chat/completions`

## 3. 日志分析（基于项目日志文件）

日志文件：`logs/eqbench3.log`

关键结论：  
1. 客户端初始化成功，且 URL 已正确归一化到 `/v1/chat/completions`。  
2. `test` 与 `judge` 两类请求都成功发出。  
3. HTTP 返回均为 `200`，说明你的本地 API 可用，且项目调用链路通。

你可以用以下命令查看最近日志：  
```powershell
Get-Content -Tail 120 logs\eqbench3.log
```

## 4. 如何启动测试

## 4.1 先做低成本连通检查（推荐）
```powershell
uv run --python .venv\Scripts\python.exe python scripts/smoke_test_api.py --type both --verbosity INFO
```

## 4.2 查看主程序参数（确认启动方式）
```powershell
uv run --python .venv\Scripts\python.exe python eqbench3.py --help
```

## 4.3 正式运行 benchmark（会消耗较多 token）
如果你的 `.env` 已配置 `TEST_MODEL_NAME/JUDGE_MODEL_NAME`，可不显式传模型名：
```powershell
uv run --python .venv\Scripts\python.exe python eqbench3.py --model-name local-api-run --no-elo --iterations 1 --threads 2
```

说明：  
- 上述命令虽然关闭了 ELO，但仍会跑全部场景（token 仍可能较高）。  
- 若你当前只想确认“项目已适配并可跑通”，建议只执行 `smoke_test_api.py`。
