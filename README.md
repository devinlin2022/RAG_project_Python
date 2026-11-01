# RAG Agent

把几十页的合同、制度、标书或汇报 PPT 变成可追问的问答服务：回答带页码引用，文档里没写的就说没有。

![version](https://img.shields.io/badge/version-1.6.0-blue)
![python](https://img.shields.io/badge/python-3.10%2B-3776AB)
![tests](https://img.shields.io/badge/tests-203%20passed-brightgreen)

单进程运行，Gradio 界面与 FastAPI 接口共用同一份索引，一条命令启动。

## 功能

- **多格式解析** —— PDF、Word、PPT、Excel、Markdown、TXT，统一成带页码的页级文本
- **混合检索** —— BM25 关键词 + bge-m3 向量，RRF 融合后精排
- **页码引用** —— 每个结论标注来源页，可回原文核对
- **双模式** —— 快速（两路检索）与增强（改写 + 扩召回 + 精排），可切换、可同题对比
- **自带评测** —— 从文档块自动出题，页码即标准答案，不依赖人工标注
- **扫描件增强** —— 扫描页 OCR、图表描述、表格转 Markdown，按页路由，只对需要的页计费
- **等待有回执** —— 解析、增强检索、出题评测都弹出遮罩：计时、阶段提示、滚动小知识，挡住重复点击
- **模型有备选** —— 对话 / 精排 / 视觉各配 2 个备选，主模型下线时自动切换；向量模型可候选实测

## RAG 技术方案

| 环节 | 实现 | 模型 / 参数 |
|---|---|---|
| 解析 | pypdfium2 取文本层，pdfplumber 只精查表格页；python-docx / python-pptx / openpyxl | — |
| 切分 | 结构感知（标题层级）为主，可选语义 / 递归 / 混合；PPT 一页一块 | 500 字，80 重叠 |
| 稀疏检索 | 自实现 BM25 | jieba 分词，拉丁 token 不拆 |
| 向量检索 | NumPy 归一化内积，精确 top-k | `BAAI/bge-m3`，1024 维 |
| 融合 | Reciprocal Rank Fusion | k = 60 |
| 精排 | 交叉编码器重排候选块 | `BAAI/bge-reranker-v2-m3` |
| 生成 | 只依据命中块作答，输出 `[n]` 引用 | `deepseek-ai/DeepSeek-V3.2` |
| 视觉 | 扫描页转录、图表与流程图描述 | `Qwen/Qwen3-VL-8B` / `32B` |

不引入 LangChain、FAISS、rank-bm25：检索链路里每个分数、每段耗时都直接可见。索引存在 SQLite（WAL），float32 向量随块存，删文档不会留下索引漂移。

PDF 与 Office 文档保留真实页码，纯文本按虚拟页切分；PPT 一张幻灯片一块，避免整份幻灯片挤成一个块、引用变成 `p.1-12`。

## 快速开始

```bash
git clone https://github.com/devinlin2022/RAG_project_Python.git
cd RAG_project_Python
pip install -r requirements.txt
cp .env.example .env          # 只需填 SILICONFLOW_API_KEY
python check_llm.py           # 校验 Key 与四条模型通道
python check_llm.py --candidates  # 逐个实测主模型与备选（每个 1 次调用）
python run.py                 # http://127.0.0.1:7860
```

首页拖入文件 → 「上传并入库」→ 开始提问。没有 `.env` 也能启动，界面里可临时粘贴 Key（只存内存）。

**Colab**：打开 [`colab.ipynb`](colab.ipynb)，在 🔑 Secrets 里填 `SILICONFLOW_API_KEY`（可选 `NGROK_AUTHTOKEN`），依次运行三格即可拿到公网地址。也可以直接：

```python
from tools.colab import boot
boot()
```

**Docker**：

```bash
docker build -t rag-agent .
docker run --rm -p 7860:7860 --env-file .env -v "$PWD/data:/app/data" rag-agent
```

## 配置

完整清单见 [`.env.example`](.env.example)。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SILICONFLOW_API_KEY` | 空 | 唯一 Key，问答 / 向量 / 精排 / 视觉共用 |
| `SILICONFLOW_BASE_URL` | `https://api.siliconflow.cn/v1` | `.cn` 与 `.com` 的 Key 和模型目录不通用 |
| `LLM_MODEL` | `deepseek-ai/DeepSeek-V3.2` | 生成模型 |
| `EMBED_MODEL` | `BAAI/bge-m3` | 向量模型 |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | 精排模型，置空则按融合分排序 |
| `OCR_ENABLED` | `true` | 仅扫描页触发，纯文字文档不产生视觉费用 |
| `LLM_FALLBACKS` 等 | 见 `.env.example` | 四条通道各 2 个备选；仅在模型下线（404 / 400-20012）时切换 |
| `EMBED_CANDIDATES` | `bge-large-zh-v1.5` 等 | 只做候选实测，不自动切换——换向量模型要重建索引 |

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| POST | `/ingest` | 上传文件，sha256 幂等 |
| POST | `/ask` | 同步问答，`mode` 取 `fast` / `enhanced` |
| POST | `/ask/stream` | SSE 流式返回流水线事件 |
| GET | `/documents` · `/system` · `/queries` | 文档列表、系统信息、最近查询 |

```bash
curl -s -X POST http://127.0.0.1:7860/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"付款账期最长是多少天？","mode":"enhanced"}'
```

Swagger 在 `/docs`。

## 评测

```bash
python -m eval.run_eval --doc demo-policy --top-k 5            # 检索指标
python -m eval.run_eval --doc demo-policy --top-k 5 --generate # 追加引用与拒答
```

仓库自带示例文档 `data/samples/demo-policy.pdf`（8 页虚构制度），21 道题：

| 模式 | hit@5 | MRR |
|---|---|---|
| 快速检索 | 100% | 0.972 |
| 增强检索 | 100% | 1.000 |

引用有效率 20/20，拒答 3/3。示例库只有 10 个块，两组数字是回归基线，不代表策略差异；换文档用界面里的「自动出题并评测」现测。

## 项目结构

```
app/     Gradio 界面 + FastAPI 路由，Flowbite 风格外壳
rag/     解析 → 切分 → 索引 → 检索 → 生成
eval/    评测集、指标、自动出题
tools/   Colab 启动、离线冒烟、示例文档生成
tests/   203 个测试
```

## 已知限制

- 中文提问 + 英文文档时 BM25 无共同 token，实际只剩向量一路，靠 bge-m3 与精排兜底。
- 扫描页 OCR 每页 15–30 秒，受 `OCR_MAX_PAGES` 与 `ENHANCE_MAX_SECONDS` 双重预算限制。
- 索引常驻内存、单进程单机；百万级块需要换 ANN 与独立向量库。
- 默认不开启鉴权，公网部署请设置 `APP_API_KEY`。

更新记录见 [CHANGELOG.md](CHANGELOG.md)。
