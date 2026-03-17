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
- 若你当前只想确认”项目已适配并可跑通”，建议只执行 `smoke_test_api.py`。

---

## 5. 离线批量测评（无 API 模型的拆分流程）

适用场景：待测模型**无法提供 API**，只能通过 `input.jsonl → 内部系统 → output.jsonl` 的批处理方式获取回答。

本流程将测评拆分为三步：
1. **导出测试输入** — 生成 `input.jsonl`
2. **内部系统推理** — 你在内部系统中运行，得到 `output.jsonl`
3. **裁判模型打分** — 本地调用 judge API 对输出进行 Rubric 评分

### 5.1 步骤一：导出 input.jsonl

```powershell
.venv\Scripts\python.exe scripts/export_input.py --output input.jsonl
```

参数说明：
| 参数 | 说明 |
|------|------|
| `--output, -o` | 输出文件路径（默认 `input.jsonl`） |
| `--iterations` | 每个场景重复次数（默认 1） |

**生成结果：** 45 条记录（每条 = 一个完整场景），每行格式：

```json
{
  “id”: “scenario_1_iter_1”,
  “scenario_id”: “1”,
  “iteration”: 1,
  “scenario_type”: “standard”,
  “num_turns”: 3,
  “has_debrief”: true,
  “prompt”: “You will participate in a multi-turn scenario with 3 turn(s)...”
}
```

**prompt 结构说明：**
- 每个场景的多轮对话被合并到一个 prompt 中
- 轮次之间用 `=== TURN N ===` 标记
- 非分析类场景末尾有 `=== DEBRIEF ===` 反思环节
- 模型需对每一轮分别作答，各轮回复之间用 `===TURN===` 分隔

### 5.2 步骤二：在内部系统中生成回答

将 `input.jsonl` 提交到内部推理系统，生成 `output.jsonl`。

**output.jsonl 格式要求（每行）：**

```json
{
  “id”: “scenario_1_iter_1”,
  “output”: “# I'm thinking & feeling\n...\n===TURN===\n# I'm thinking & feeling\n...\n===TURN===\n...”
}
```

关键要求：
- **`id` 字段必须保留**，与 input.jsonl 中的 id 一一对应
- **`output` 字段**包含模型对该场景所有轮次的回复
- 各轮回复之间用 `===TURN===` 分隔（独占一行）
- 如场景有 3 轮 + debrief，则 output 中应有 4 段回复，用 3 个 `===TURN===` 分隔

**output 示例（3 轮标准场景 + debrief）：**

```
# I'm thinking & feeling

我对这个情况感到焦虑...

# They're thinking & feeling

他们似乎很困惑...

# My response

我理解你的担忧，让我解释一下...
===TURN===
# I'm thinking & feeling

事态有了新进展...

# They're thinking & feeling

她看起来有些不满...

# My response

我觉得我们需要直接沟通...
===TURN===
# I'm thinking & feeling

面对指控我感到很委屈...

# They're thinking & feeling

老板显然很不耐烦...

# My response

我可以证明不是我做的...
===TURN===
回顾这次对话，我注意到几个可以改进的地方...
```

### 5.3 步骤三：裁判模型打分

确保 `.env` 中已配置 judge 模型的 API：

```ini
JUDGE_API_KEY=sk-...
JUDGE_API_URL=https://api.openai.com/v1/chat/completions
JUDGE_MODEL_NAME=anthropic/claude-3.7-sonnet
```

运行打分：

```powershell
.venv\Scripts\python.exe scripts/judge_output.py `
    --input input.jsonl `
    --output output.jsonl `
    --judge-model anthropic/claude-3.7-sonnet `
    --model-name my-internal-model `
    --result-file my_results.json `
    --threads 4
```

参数说明：
| 参数 | 说明 |
|------|------|
| `--input, -i` | export_input.py 生成的 input.jsonl |
| `--output, -o` | 内部系统生成的 output.jsonl |
| `--judge-model` | 裁判模型 API 标识（默认读取 `.env` 中 `JUDGE_MODEL_NAME`） |
| `--model-name` | 待测模型的逻辑名称，用于结果展示（默认 `batch-model`） |
| `--result-file` | 评分结果保存路径（默认 `eqbench3_runs.json`） |
| `--threads` | 并发调用 judge API 的线程数（默认 4） |

**输出示例：**

```
============================================================
        EQBench3 Rubric Score Summary
============================================================
  Model:          my-internal-model
  Judge:          anthropic/claude-3.7-sonnet
  Tasks scored:   45 / 45
  Rubric (0-20):  15.40
  Rubric (0-100): 77.00
============================================================

Results saved to my_results.json
```

### 5.4 注意事项

1. **模型会预读后续轮次：** 扁平化后模型会看到所有未来的用户消息（原始测试中是逐轮接收的），可能轻微影响回复策略。用相同方法测试不同模型仍可进行有效的相对比较。

2. **分隔符容错：** 如果模型未严格使用 `===TURN===` 分隔，脚本会尝试 `---TURN---`、`=== TURN N ===` 等备选模式自动匹配。

3. **生成参数：** 原始测试默认使用 `temperature=0.7, max_tokens=8000`。如果内部系统使用不同参数，可能影响结果。

4. **仅 Rubric 评分：** 此流程只支持 Rubric 打分（0-100），不支持 ELO 对比排名。
