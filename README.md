# rag-agent · 文档助手

把一个几十页的文件（合同、制度、投标书、汇报 PPT）变成能追问、能给出处、会说"没写"的问答服务。
单进程：Gradio 界面 + FastAPI 接口共享同一份索引，一条命令启动。

> 代码注释用英文，界面文案用中文。仓库只带虚构示例，不含任何真实业务材料。

## 两件事，两个模式

界面只有两个页签，因为它们对应两个真实动作：

**① 文档与分块** — 上传 → 解析 → 切分 → 建索引 → **看到每一个块**。
上方的页轴显示全文字量分布，下方每个块按字数画成条形、按类型着色；问答后被引用的块标红。

**② 提问** — 同一个问题，两种检索强度：

| 模式 | 做了什么 | 适合 |
|---|---|---|
| **快速检索** | 向量 + BM25 → RRF 融合 → 直接作答，约 1 秒出首字 | 大多数日常提问 |
| **增强检索** | 先改写追问，再扩大到 36 个候选并做交叉编码器精排 | 追问、编号/金额类精确问题 |

两种模式都会把后台动作逐条显示出来：改写 → 向量化（维度、L2 范数）→ 向量检索（余弦、最高分）
→ BM25（分词结果）→ 融合（候选数）→ 精排（前后对比）→ 生成（首字延迟、token 数）。

## 支持的格式

| 格式 | 抽取方式 | 页码含义 |
|---|---|---|
| PDF | pdfium 取文本层；表格页再用 pdfplumber 精查 | 真实页 |
| Word `.docx` | python-docx，表格转 Markdown，识别分页符 | 真实页或虚拟页 |
| PPT `.pptx` | python-pptx，**连接线还原成 `A -> B` 箭头关系** | 一张幻灯片 = 一页 |
| Excel `.xlsx` / `.csv` | openpyxl / csv，每个工作表转 Markdown 表 | 表内分虚拟页 |
| Markdown / TXT / HTML | 按空行切块后打包 | 虚拟页 |

幻灯片是特例：拆块时**一页一块**，否则整份 PPT 会被塞进一个块、引用变成 `p.1-12`。

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env       # 只填 SILICONFLOW_API_KEY 这一行
python check_llm.py        # 校验 Key、四个模型 ID 与四条通道
python run.py              # http://127.0.0.1:7860
```

没有 `.env` 也能启动：界面「运行状态与用量」里可以直接粘贴 Key（只存内存），解析和关键词检索不花钱。

在 Colab 上：

```python
!pip install -r requirements.txt
import os
os.environ["SILICONFLOW_API_KEY"] = "sk-..."
os.environ["NGROK_AUTHTOKEN"] = "..."      # 填了就自动开隧道，无需 NGROK_ENABLED
!python run.py
```

`APP_HOST` 会自动切到 `0.0.0.0`（检测到 Colab 或 ngrok），不需要手动改。

## 配置

一个入口，四个模型，全在硅基流动：

| 变量 | 默认 | 说明 |
|---|---|---|
| `SILICONFLOW_API_KEY` | 空 | 唯一的 Key；问答、向量、精排、视觉共用 |
| `SILICONFLOW_BASE_URL` | `https://api.siliconflow.cn/v1` | `.cn` 与 `.com` 的 Key 和模型目录不通用 |
| `LLM_MODEL` | `deepseek-ai/DeepSeek-V3.2` | 换模型只改这一行 |
| `EMBED_MODEL` | `BAAI/bge-m3` | 1024 维 |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | 关闭后按融合分排序 |
| `VISION_MODEL` | `Qwen/Qwen3-VL-32B-Instruct` | OCR 与图注 |
| `OCR_ENABLED` | `false` | 开启才会计费；`OCR_CONCURRENCY=3` 并发 |
| `PARSE_WORKERS` / `EMBED_CONCURRENCY` | `4` / `3` | 多文件并行解析、多批并行向量化 |
| `PRICE_IN_PER_M` / `PRICE_OUT_PER_M` | `0` | 填了就在用量里折算金额 |

旧变量名（`LLM_API_KEY` / `LLM_BASE_URL`）仍然生效。完整清单见 `.env.example`。

## 速度

同一台机器、48 页 PDF、冷缓存：

| 环节 | 之前 | 现在 |
|---|---|---|
| 解析 | 911 ms（pdfminer 全文 + pdfplumber 逐页找表） | **357 ms**（pdfium + 只精查像表格的页） |
| 扫描件增强 | 逐页串行，20 页约 4 分钟 | 并发 3，约 1 分钟 |
| 建索引 | 单批串行向量化 | 分批并发，块多时近似线性加速 |

解析快的关键是**不再对每一页跑表格识别**：先用 pdfium 拿到文本与图元数，
只有"有网格痕迹 / 多列文本"的页才交给 pdfplumber。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 免鉴权健康检查 |
| POST | `/ingest` | 上传任意支持格式（sha256 幂等） |
| GET | `/documents` · `/documents/{id}/pages` | 文档列表、页级信号 |
| POST | `/documents/{id}/index` · `/enhance` | 建索引、跑增强识别 |
| POST | `/ask` | 同步问答；`mode` 取 `fast` / `enhanced` |
| POST | `/ask/stream` | SSE：每条流水线事件一帧 |
| GET | `/system` · `/queries` | 系统信息、最近查询（Swagger 在 `/docs`） |

```bash
curl -s -X POST http://127.0.0.1:7860/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"付款账期最长是多少天？","mode":"enhanced"}'
```

## 评测

`eval/golden.jsonl` 每题带标注页码，`answerable:false` 的题检验拒答。

```bash
python -m eval.run_eval --doc demo-policy --top-k 5
python -m eval.run_eval --doc demo-policy --top-k 5 --generate   # 追加引用与拒答指标
```

仓库自带 `data/samples/demo-policy.pdf`（8 页虚构制度，含一张权限矩阵、一页流程图）：
入库 1,967 字符，增强后 2,433 字符，10 块；21 题 hit@5 100%、MRR 1.000、引用 20/20、拒答 3/3。
块太少，三路检索都是满分 —— 这是**回归基线**，不是策略对比。

## 架构

```
文件 ─┬─ pdfium / docx / pptx / xlsx 抽取（并行） ─┐
      ├─ pdfplumber 表格 → Markdown（仅可疑页）    ├─ 结构感知切分 ─┬─ BM25（jieba + 拉丁 token 保护）
      ├─ 视觉模型 OCR（扫描页，并发）              │                 └─ bge-m3 向量（精确余弦）
      └─ 视觉模型图注（流程图箭头）                ┘                 ↓
                                            RRF(k=60) → 精排 → 带 [n] 引用作答
```

| 层 | 选型 | 为什么 |
|---|---|---|
| 切分 | 自实现 structured / hybrid / semantic / recursive + 按页 | 可解释、可测、毫秒级 |
| 稀疏 | 自实现 BM25（40 行） | 省一个依赖，词表与块同生命周期 |
| 向量 | NumPy 归一化内积 | 小库精确搜索，结果等同 `IndexFlatIP` |
| 融合 | RRF | 余弦 [0,1] 与 BM25 [0,40] 量纲不同，只合并名次 |
| 存储 | SQLite（WAL），float32 向量随块存 | 删文档就是一条 DELETE，不会索引漂移 |
| 服务 | FastAPI + Gradio `mount_gradio_app` | 单进程单状态，界面与接口共用索引 |

不用 LangChain / FAISS / rank-bm25：这条链路的价值就在分数与耗时可见，框架会把它们藏起来。

## 数据边界

- 仓库只带虚构示例：`data/samples/` 下 6 个文件全部由 `python tools/make_demo_doc.py` 从同一份正文生成（pdf 8 页 / docx / pptx 4 张幻灯片 / xlsx 2 个表 / md / 英文 3 页 fixture），所以二进制可复现，改文案就能重新出全套。
- `data/uploads/`、`data/rag.sqlite`、`eval/report.md` 在 `.gitignore` 里：上传件与查询日志只留在本机。
- 清空 `data/` 即可分发，服务启动会重建空库。

## 已知限制

- 中文提问 + 英文文档时 BM25 失效（无共同 token），实际只剩向量一路，靠 bge-m3 与精排兜底。
- 扫描页 OCR 每页 15–30 秒，受 `OCR_MAX_PAGES` 与 `ENHANCE_MAX_SECONDS` 双重预算限制。
- 索引常驻内存，单机单进程；百万级块要换 ANN 与独立向量库。
- Excel 只取单元格文本，图表与图片不解析。
