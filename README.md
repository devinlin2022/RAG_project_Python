# rag-agent

Online RAG assistant based on Gradio, SiliconFlow / Ollama

一个按页抽取、可逐步扩展的文档问答服务。当前为 **V1**：完成「PDF → 按页抽取 → 页级标记 → 提示词 → 回答」的最小闭环，并为后续版本建立页级元数据。

页面与接口由同一进程提供，共享同一份文档状态。

> 代码约定：代码中的注释与 docstring 使用英文；界面文案与错误提示使用中文。

---

## 本版范围

| 能力          | 状态  | 说明                                      |
| ----------- | --- | --------------------------------------- |
| PDF 按页抽取    | 已实现 | `pdfminer.extract_pages`，保留页码           |
| 页级质量标记      | 已实现 | `scan_like`（疑似扫描页）与 `table_like`（疑似表格页） |
| 页级问答        | 已实现 | 人工选页，回答仅基于所选页面                          |
| 引用标注        | 已实现 | 提示词要求 `[第N页]`                           |
| 文本切分        | 未实现 | V2 引入语义切分                               |
| 向量索引 / BM25 | 未实现 | V2 / V3                                 |
| 重排          | 未实现 | V2                                      |
| 流式输出        | 未实现 | V4                                      |
| 持久化         | 未实现 | V5                                      |

本版**不含检索**：问答需要显式选择页码。这一限制是刻意的——它使文档到回答的链路可以在没有索引的情况下被独立验证，同时明确 V2 要自动化的具体对象。

---

## 环境配置（硅基流动 DeepSeek）

### 1. 获取 API Key

1. 注册并登录 [硅基流动控制台](https://cloud.siliconflow.cn)
2. 打开「API 密钥」页面（<https://cloud.siliconflow.cn/account/ak），新建密钥>
3. 复制以 `sk-` 开头的完整字符串（只显示一次）

### 2. 创建配置文件

```bash
cp .env.example .env          # Windows: copy .env.example .env
```

`.env` 位于项目根目录，已被 `.gitignore` 忽略，不会提交到仓库。

### 3. 填写三个核心变量

| 变量             | 硅基流动的取值                         | 说明                       |
| -------------- | ------------------------------- | ------------------------ |
| `LLM_BASE_URL` | `https://api.siliconflow.cn/v1` | **必须包含末尾的 `/v1`**，否则 404 |
| `LLM_API_KEY`  | `sk-xxxxxxxxxxxxxxxx`           | 控制台复制的完整密钥，不要带空格或引号      |
| `LLM_MODEL`    | `deepseek-ai/DeepSeek-V3`       | 模型广场中的**完整模型 ID**，不是显示名称 |

填写后的 `.env` 示例：

```dotenv
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL=deepseek-ai/DeepSeek-V3

TEMPERATURE=0.2
MAX_TOKENS=1024
LLM_TIMEOUT=90
```

`LLM_BASE_URL` 与 `LLM_MODEL` 都在 `.env` 中配置，代码里没有硬编码，换模型只需改这一行。

### 4. DeepSeek 模型选择

| `LLM_MODEL` 候选            | 适用场景       | 说明                     |
| ------------------------- | ---------- | ---------------------- |
| `deepseek-ai/DeepSeek-V3` | 通用文档问答（默认） | 速度快、成本低，适合抽取型问答        |
| `deepseek-ai/DeepSeek-R1` | 需要显式推理链    | 会额外返回推理内容，首 token 延迟更高 |
| 其他 `deepseek-ai/*`        | 平台上线新版本后   | 按模型广场实际 ID 填写          |

模型 ID 随平台更新而变动，**以控制台模型广场为准**。本项目不预设枚举，运行时也不需要改动代码，只改 `.env` 的 `LLM_MODEL`。

### 5. 验证配置

```bash
python check_llm.py
```

该脚本依次完成三件事：打印当前配置、拉取平台模型列表并高亮 DeepSeek 相关 ID、发送一次 `ping` 对话请求。

正常输出形如：

```
== 当前配置 ==
LLM_BASE_URL : https://api.siliconflow.cn/v1
LLM_MODEL    : deepseek-ai/DeepSeek-V3
LLM_API_KEY  : sk-abc…

平台可用模型 92 个，其中 DeepSeek 相关 4 个：
  - deepseek-ai/DeepSeek-R1
  - deepseek-ai/DeepSeek-V3  ← 当前配置
  ...

对话测试成功，返回：'pong'
配置可用，可以启动：python run.py
```

若 `LLM_MODEL` 不在返回列表中，脚本会直接提示，不必等到页面上报错。

### 6. 常见错误对照

| 现象                    | 原因                                 | 处理                                         |
| --------------------- | ---------------------------------- | ------------------------------------------ |
| `API Key 无效（401）`     | 密钥不完整，或复制时带入空格                     | 重新复制，确保 `sk-` 开头且无换行                       |
| `接口或模型不存在（404）`       | `LLM_BASE_URL` 缺 `/v1`，或模型 ID 拼写错误 | 核对 base_url；用 `check_llm.py` 列出 ID         |
| `连接超时` / `ProxyError` | 代理或 VPN 拦截                         | 关闭代理，或将 `api.siliconflow.cn` 加入 `NO_PROXY` |
| `HTTP 400`            | 模型不支持请求中的某个字段                      | 本项目只发送标准字段；若仍报错请贴出响应体                      |
| 页面提示「未配置 LLM_API_KEY」 | `.env` 未创建或变量名为空                   | 确认文件在项目根目录、变量拼写正确                          |

修改 `.env` 后需要重启服务才会生效。

---

## 启动

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python check_llm.py                # 先验证配置
python run.py                      # 再启动服务
```

- 页面：<http://127.0.0.1:7860>
- 接口文档：<http://127.0.0.1:7860/docs>

在 PyCharm 中把运行目标设为 `run.py`，工作目录设为项目根目录即可。

仓库自带 `data/samples/sample-3pages.pdf`（正文页 / 表格页 / 空白页），可用于验证三项标记是否生效。二进制测试文件不纳入版本库，克隆后可用任意 PDF 代替。

---

## 页面操作

### 文档页

| 操作              | 预期结果                                |
| --------------- | ----------------------------------- |
| 上传 PDF → 点击「解析」 | 显示页数、字符数、耗时；页级表按页列出字符、密度、表格数与两项标记   |
| 点击表格任意行         | 右侧显示该页原文，首行为页码与统计                   |
| 勾选「仅显示可疑页」      | 表格仅保留 `scan_like` 或 `table_like` 的行 |
| 使用「查看页码」下拉      | 单页原文的备用入口（表格点击无响应时使用）               |

### 问答页

| 操作                | 预期结果            |
| ----------------- | --------------- |
| 下拉框多选页码 + 输入问题    | 回答仅基于所选页面，并标注页号 |
| 「补充页码范围」填 `1-5,8` | 与下拉选择合并，便于批量选择  |
| 点击「填入可疑页」         | 自动选中所有疑似扫描页与表格页 |
| 未选页码直接提问          | 提示至少选择一页        |
| 所选页面字符数超限         | 提示减少页数，不调用模型    |

---

## 接口

```bash
curl http://127.0.0.1:7860/health

curl -X POST http://127.0.0.1:7860/ingest -F "file=@data/samples/sample-3pages.pdf"

curl http://127.0.0.1:7860/docs/{doc_id}/pages

curl -X POST http://127.0.0.1:7860/ask-by-pages \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"xxx","pages":[1,2],"question":"审批由谁负责？"}'
```

`/health` 会返回当前的 `llm_base_url` 与 `llm_model`，便于确认加载的是哪份配置。

---

## 目录结构

```
rag-agent/
├── run.py                  启动入口（PyCharm 运行目标）
├── check_llm.py            环境与模型连通性检查
├── app/
│   ├── main.py             FastAPI + 挂载 Gradio
│   ├── api.py              四个接口
│   ├── ui.py               两个标签页与事件绑定
│   └── state.py            进程内文档存储
├── rag/
│   ├── config.py           pydantic-settings，启动即校验
│   ├── utils.py            页码表达式解析等无依赖工具
│   ├── ingest/loader.py    按页抽取 + 页级质量标记
│   └── generate/
│       ├── prompt.py       系统提示与上下文拼装
│       ├── llm.py          OpenAI 兼容调用与错误映射
│       └── answer.py       页级问答（界面与接口共用）
├── tests/
├── data/{samples,uploads}/
├── .env.example
└── requirements.txt
```

---

## 全部配置项

| 变量                                                | 默认值                             | 说明                     |
| ------------------------------------------------- | ------------------------------- | ---------------------- |
| `LLM_BASE_URL`                                    | `https://api.siliconflow.cn/v1` | OpenAI 兼容端点，末尾需含 `/v1` |
| `LLM_API_KEY`                                     | 空                               | 为空时问答返回配置提示            |
| `LLM_MODEL`                                       | `deepseek-ai/DeepSeek-V3`       | 平台模型 ID                |
| `TEMPERATURE`                                     | `0.2`                           | 抽取型问答取低值               |
| `MAX_TOKENS`                                      | `1024`                          | 单次回复上限                 |
| `LLM_TIMEOUT`                                     | `90`                            | 请求超时（秒）                |
| `SCAN_PAGE_CHAR_THRESHOLD`                        | `200`                           | 单页字符数低于此值判为疑似扫描页       |
| `TABLE_COLUMN_RATIO`                              | `0.4`                           | 列结构行占比达到此值判为疑似表格页      |
| `MAX_CONTEXT_CHARS`                               | `12000`                         | 页级问答的上下文字符上限           |
| `CHUNK_STRATEGY` / `CHUNK_SIZE` / `CHUNK_OVERLAP` | `semantic` / `500` / `80`       | V2 起使用，本版仅占位           |

配置在启动时由 pydantic 校验，类型错误会直接失败；`extra="ignore"`，向 `.env` 追加未声明变量不会报错。

---

## 测试

```bash
pytest -q
```

- `tests/test_pages.py`：页码范围解析，不依赖第三方包
- `tests/test_loader.py`：页级标记为纯函数测试；PDF 抽取相关测试在未安装依赖时自动跳过

将任意 PDF 复制为 `tests/sample.pdf` 后，可额外验证页号连续性。

---

## 已知限制

- 无检索能力，问答依赖人工选页，受 `MAX_CONTEXT_CHARS` 限制
- 扫描版 PDF 无文本层，仅能在页级表中被标记，需 OCR，不在本期范围
- 多栏排版可能出现阅读顺序错乱，`detect_vertical=True` 只能部分缓解
- 表格仅做检测与标记，不抽取结构（V2 以行级切分形式实现）
- 单文档、单进程、内存态，重启后需重新解析
- 无鉴权，仅建议绑定 `127.0.0.1` 本地使用

---

## 版本路线

| 版本 | 主题      | 关键交付                   |
| -- | ------- | ---------------------- |
| V1 | 骨架与按页抽取 | 本版                     |
| V2 | 语义切分与精排 | 语义断点、表格旁路、FlatIP、交叉编码器 |
| V3 | 混合检索与引用 | BM25、RRF、可核验引用         |
