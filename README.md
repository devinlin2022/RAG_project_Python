# rag-agent

Online RAG assistant based on Gradio, SiliconFlow / Ollama

按页抽取、按结构切分的文档问答服务。当前 **V2**。

> 代码注释与 docstring 用英文；界面文案与错误提示用中文。

---

## 项目目的

合同、制度这类文档不适合按字数切：一条被切成两个半句，两个半句都不能当证据。同时有三类页面的内容根本进不了文本层——扫描页没有字，表格被压成一行，流程图的含义在箭头里。

V2 解决的就是这两件事：**把文档读对，再把它切对**。检索留到 V3，因为现在还分不清问题是「没召回」还是「切错了」。

适用：合同、协议、制度、规范、含表格的清单。不适用：Word / Excel、纯图纸。

---

## 本版变更（V2）

| 能力 | V1 | V2 |
| --- | --- | --- |
| 表格 | 只检测、只标记 | 输出 Markdown，表头在每个片段重复 |
| 扫描页 | 只标记 | OCR 通道转录 |
| 图形页 | 无法处理 | 图注通道，输出 `A -> B` 箭头关系 |
| 切分 | 无 | 混合 / 结构化 / 语义 / 递归四种 |
| 策略选择 | 无 | 读编号结构自动推荐，毫秒级返回 |
| 质量证据 | 无 | 五项结构指标（含类型保留率） |
| 界面 | 2 个标签页 | 4 个：文档 / 增强识别 / 切分策略 / 问答 |
| 接口 | 4 个 | 7 个 |
| 测试 | 13 | 45 passed, 1 skipped |

---

## 技术架构

```
PDF ──▶ 按页抽取 ──▶ 逐页路由 ──┬─▶ 文本层
       (pdfminer)    (本地判定)  ├─▶ 表格 → Markdown（pdfplumber，离线）
                                ├─▶ OCR（视觉模型）
                                └─▶ 图注（视觉模型，读箭头）
                                      │
                                      ▼
                          结构普查 → 推荐策略 → 四策略切分
                                      │
                                      ▼
                    Chunk = 文本 + 页码 + 章节路径 + 类型
```

| 层 | 选型 | 说明 |
| --- | --- | --- |
| 文本层抽取 | `pdfminer.six` `extract_pages` | 比 `extract_text` 慢约 25%，保留页码 |
| 表格结构 | `pdfplumber` | 有框线取网格；无框线从文本解析管道/空格对齐表格 |
| 页面渲染 | `pypdfium2` | 纯 wheel，无系统依赖 |
| OCR / 图注 | OpenAI 兼容视觉接口 | 默认 `Qwen3-VL-32B`，换平台只改 base_url |
| 向量 | `bge-m3` | 语义断点用，1024 维 |
| 编号解析 | 自研正则 | 支持 `第X条`、`一、`、`（一）`、`2.1` 两套体系 |
| 切分 | 自研四策略 | 保证单条不截断、块带章节路径 |
| 服务 | `FastAPI` + `Gradio` 单进程 | 页面与接口共享同一份文档状态 |
| 配置 | `pydantic-settings` | 启动即校验 |

不用 LangChain：它的 splitter 表达不了「按第X条切、条款不许截断、块带章节路径」。

---

## 环境配置

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

`.env` 已 gitignore，不会进仓库。只需填 `LLM_API_KEY`，其余可用默认值。

**必需**

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `LLM_BASE_URL` | `https://api.siliconflow.cn/v1` | 末尾必须带 `/v1` |
| `LLM_API_KEY` | 空 | 以 `sk-` 开头；`.cn` 与 `.com` 的 Key 不通用 |
| `LLM_MODEL` | `deepseek-ai/DeepSeek-V3` | 完整模型 ID，不是显示名 |

**OCR / 图注**（默认关闭）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `OCR_ENABLED` | `false` | 打开后才允许调用视觉模型 |
| `OCR_MODEL` | `Qwen/Qwen3-VL-32B-Instruct` | 表格输出 Markdown；`PaddleOCR-VL` 只出纯文本，不推荐 |
| `VISION_MODEL` | `Qwen/Qwen3-VL-32B-Instruct` | 读箭头关系 |
| `OCR_RENDER_DPI` | `150` | 小字用 200 |
| `OCR_MAX_PAGES` | `20` | 单次最大调用页数 |
| `OCR_MIN_INTERVAL` | `1.5` | 两次调用最小间隔（秒），限速用 |
| `ENHANCE_MAX_SECONDS` | `120` | 整次增强的时间上限 |

**向量 / 切分**

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `EMBED_MODEL` | `BAAI/bge-m3` | 语义切分用；留空复用 LLM 的 Key |
| `CHUNK_STRATEGY` | `auto` | `auto` / `hybrid` / `structured` / `semantic` / `recursive` |
| `CHUNK_SIZE` | `500` | `auto` 时在 300–900 间浮动 |
| `CHUNK_OVERLAP` | `80` | 结构化切分实际取 0 |
| `SEMANTIC_PERCENTILE` | `85` | 语义断点分位数 |

`OCR_BASE_URL` / `OCR_API_KEY` / `EMBED_BASE_URL` / `EMBED_API_KEY` 留空时自动复用 `LLM_BASE_URL` / `LLM_API_KEY`。

**验证**（别等到页面上报错）

```bash
python check_llm.py          # 问答 + 向量
python check_llm.py --ocr    # 再加一次视觉调用（会计费）
```

会列出四类模型是否在当前站点存在，并实测连通性。`MISS` = 模型 ID 不存在，`401` = Key 不属于这个站点，`402` = 余额不足。

---

## 启动

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python check_llm.py                # 先验配置
python run.py
```

- 页面 <http://127.0.0.1:7860>
- 接口文档 <http://127.0.0.1:7860/docs>

在 PyCharm 把运行目标设为 `run.py`，工作目录设为项目根。

需要临时分享时在 `.env` 打开 `NGROK_ENABLED=true` 并填 `NGROK_AUTHTOKEN`，控制台会打印公网链接；隧道失败只警告，本地照常可用。

---

## 后续版本

| 版本 | 主题 | 关键交付 |
| --- | --- | --- |
| V1 | 骨架与按页抽取 | 已完成 |
| V2 | 文档理解与切分 | 本版 |
| V3 | 混合检索与精排 | BM25 + FAISS、RRF 融合、cross-encoder 重排、页码+条款双定位引用 |
| V4 | 真流式与多轮 | SSE、TTFT、查询改写 |
| V5 | 持久化与文档管理 | SQLite、索引持久化、幂等入库 |
| V6 | 评测与调参 | 标注问题集、hit@5 / MRR、A/B 实验 |
| V7 | 硬化与发布 | 鉴权、限流、Docker、v1.0.0 |
