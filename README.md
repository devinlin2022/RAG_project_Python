# rag-agent · PDF 助手

把一个几十页的 PDF（合同、制度、投标文件、会议材料）变成能追问、能给出处、会说"没写"的问答服务。
单进程：Gradio 界面 + FastAPI 接口共享同一份索引，一条命令启动。

> 代码注释与 docstring 用英文；界面文案与错误提示用中文。

## 这一版能做什么

| 页签 | 你能看到什么 |
|---|---|
| 文档 | 多文件上传、文档列表与删除、页级信号表（字符/表格/扫描/图形）、单页原文 |
| 文档 › 增强识别 | 逐页路由判定、扫描页 OCR、表格结构化、图形页图注；结果入库，重启不重跑 |
| 切分与索引 | 结构普查与策略推荐、四种策略指标对比、块列表、一键建索引 |
| 检索调试 | 同一问题分别走向量 / BM25 / RRF，再对照精排前后，逐块查看分数 |
| 智能问答 | 流式回答、编号引用、改写后的检索问题、分阶段耗时 |
| 评测 | hit@k / MRR / 引用有效率 / 拒答准确率，结果可导出 Markdown |
| 系统信息 | 版本、模型、索引规模、只读配置、最近查询 |

## 技术架构

```
PDF ─┬─ pdfminer 按页文本 ──────────────┐
     ├─ pdfplumber 表格 → Markdown       ├─ 结构感知切分 ─┬─ BM25（jieba + 拉丁 token 保护）─┐
     ├─ 视觉模型 OCR（扫描页）           │                  └─ bge-m3 向量（精确内积）─────────┤
     └─ 视觉模型图注（流程图箭头关系）   ┘                                                   │
                                                          RRF 融合（k=60）→ bge-reranker-v2-m3 精排
                                                                        → DeepSeek 带 [n] 引用作答
```

| 层 | 选型 | 为什么 |
|---|---|---|
| 抽取 | pdfminer（文本）+ pdfplumber（表格/版面信号） | 一次 pdfplumber 扫描算出表格数、图片覆盖率、矢量图元数 |
| 切分 | 自实现：structured / hybrid / semantic / recursive | 可解释、可测、毫秒级；不引入黑盒 |
| 稀疏检索 | 自实现 BM25（`rag/index/bm25.py`，40 行） | `rank-bm25` 只省 40 行，却多一个依赖和一次语料复制 |
| 向量检索 | NumPy 归一化矩阵内积（`rag/index/dense.py`） | 小库精确搜索，结果与 `IndexFlatIP` 一致，省掉 FAISS |
| 融合 | RRF | 余弦 [0,1] 与 BM25 [0,40] 量纲不同，排序融合不需要编权重 |
| 精排 | `BAAI/bge-reranker-v2-m3`（托管 `/v1/rerank`） | 真交叉编码器，不下载 2 GB torch 权重 |
| 生成 | DeepSeek-V3（OpenAI 兼容） | 一处封装，可换模型 |
| 存储 | SQLite（WAL），向量以 float32 BLOB 随块存储 | 删文档就是一条 DELETE，不存在索引文件与库不同步 |
| 服务 | FastAPI + Gradio `mount_gradio_app` | 单进程单状态，界面与接口用同一份索引 |
| 测试 | pytest，113 项 | 离线可跑，不花一分钱 |

三个取舍值得单独说明：

1. **不用 LangChain。** 这条链路每一步都要看见分数和耗时（页面就是靠这个做验收），框架的默认行为会把检索分数、切分边界藏起来。
2. **不用 FAISS / rank-bm25。** 两者在这里的功能只有几十行，换成自实现后依赖更少，且向量与块同表存储，删除不会留下孤儿索引。
3. **精排走托管接口而不是本地模型。** 本地 `bge-reranker-v2-m3` 要 torch + 2 GB 权重，CPU 上每题数百毫秒；托管是每题一次请求，延迟可见、可关。

## 环境配置

**必填**（缺了只能看文档页，不能问答）

| 变量 | 说明 |
|---|---|
| `LLM_BASE_URL` | 以 `/v1` 结尾；`.cn` 与 `.com` 是两套 Key 和模型目录 |
| `LLM_API_KEY` | `sk-` 开头 |
| `LLM_MODEL` | 平台模型广场的完整 ID，如 `deepseek-ai/DeepSeek-V3` |

**检索与精排**

| 变量 | 默认 | 说明 |
|---|---|---|
| `EMBED_MODEL` | `BAAI/bge-m3` | 1024 维；未配置 Key 时索引只有 BM25，界面会标注降级 |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | 关闭后按融合分排序 |
| `RETRIEVE_TOP_K` | `6` | 回答引用的块数 |
| `RERANK_CANDIDATES` | `24` | 送精排的候选数，最终截到 top_k |
| `RRF_K` | `60` | 融合常数 |
| `HISTORY_TURNS` / `REWRITE_ENABLED` | `3` / `true` | 多轮改写 |

**增强识别（按页计费，默认关闭）**

| 变量 | 默认 | 说明 |
|---|---|---|
| `OCR_ENABLED` | `false` | 打开后「增强识别」按钮才真正调模型 |
| `OCR_MODEL` | `Qwen/Qwen3-VL-32B-Instruct` | 表格页必须能输出 Markdown，`PaddleOCR-VL` 只吐纯文本 |
| `VISION_MODEL` | 同上 | 图注要读箭头关系，OCR 做不到 |
| `OCR_MAX_PAGES` / `ENHANCE_MAX_SECONDS` | `20` / `600` | 双重预算，超出的页跳过并在表格里标注 |

**硬化（默认不启用）**

| 变量 | 说明 |
|---|---|
| `APP_API_KEY` | 非空则所有写接口要求 `X-API-Key` |
| `UI_USERNAME` / `UI_PASSWORD` | `UI_PASSWORD` 非空则界面出现登录页 |
| `RATE_LIMIT_PER_MIN` | 按客户端 IP 的滑动窗口限流 |
| `LOG_JSON` | JSON 日志，`sk-***` 与 `Bearer` 自动脱敏 |

完整清单见 `.env.example`。

## 启动

```bash
pip install -r requirements.txt
cp .env.example .env        # 填入 LLM_API_KEY
python check_llm.py         # 校验模型 ID 与四个通道
python run.py               # http://127.0.0.1:7860
```

首次使用顺序：文档页上传 PDF（勾选自动建索引）→ 需要表格/扫描件时运行增强识别（完成后自动重建索引）→ 智能问答。

无界面用法：

```bash
python -m rag.cli ingest data/samples/demo-policy.pdf --index   # 仓库自带的虚构示例文档
python -m rag.cli ask "付款账期最长是多少天？" --doc demo-policy
python -m rag.cli eval --doc demo-policy --top-k 5
```

Docker：

```bash
docker build -t rag-agent .
docker run -p 7860:7860 --env-file .env -v "$PWD/data:/app/data" rag-agent
```

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 免鉴权健康检查 |
| POST | `/ingest` | 上传 PDF（sha256 幂等） |
| GET | `/documents` · `/documents/{id}/pages` | 文档列表、页级信号 |
| POST | `/documents/{id}/index` · `/documents/{id}/enhance` | 建索引、跑增强识别 |
| POST | `/retrieve` | 三种检索策略并排返回 |
| POST | `/ask` | 同步问答，返回答案、来源、分阶段耗时 |
| POST | `/ask/stream` | SSE：`prepared` / `delta` / `done` / `error` |
| POST | `/eval/run` | 跑 golden set |
| GET | `/system` · `/queries` | 系统信息、最近查询（Swagger 在 `/docs`） |

`/ask` 示例：

```bash
curl -s -X POST http://127.0.0.1:7860/ask \
  -H 'Content-Type: application/json' -H 'X-API-Key: <你的 APP_API_KEY>' \
  -d '{"question":"付款账期最长是多少天？","top_k":6,"strategy":"rrf"}'
```

## 评测

`eval/golden.jsonl` 每题带标注页码，`answerable:false` 的题用来检验拒答。
默认那份对应仓库自带的 `data/samples/demo-policy.pdf`（8 页虚构制度文件，21 题）。
换成自己的文档时，把同样格式的问答对写进 jsonl，再按文件名过滤：

```bash
python -m eval.run_eval --doc demo-policy --top-k 5            # 纯检索，几十秒
python -m eval.run_eval --doc demo-policy --top-k 5 --generate # 追加引用有效率与拒答准确率
```

| 指标 | 定义 |
|---|---|
| hit@k | 前 k 个块的页码命中任一标注页 |
| MRR | 第一个命中块名次的倒数 |
| 引用有效率 | 回答中的 `[n]` 落在实际来源编号内的比例 |
| 拒答准确率 | 不可答题中判为「资料未提及」的比例 |

## 示例文档实测（8 页虚构制度文件）

`data/samples/demo-policy.pdf` 由 `tools/make_demo_doc.py` 生成：8 页、含条款编号、一张审批权限表、一页矢量流程图。

| 环节 | 结果 |
|---|---|
| 入库 | 8 页 / 1,967 字符 |
| 增强识别 | 表格通道恢复 p.4 审批权限矩阵，图注通道在 p.5 还原 8 个节点与箭头关系；1 次视觉调用、14.8 s，正文 1,967 → 2,433 字符 |
| 切分与索引 | hybrid，10 块，1024 维，向量化 0.6 s |
| 检索（21 题，18 可答） | RRF + 精排 hit@5 **100%**、MRR **1.000** |
| 生成 | 引用有效率 100%（20 处引用），3 道不可答题全部拒答 |
| 单题时延 | 向量化 0.36 s、精排 0.44 s、生成 7.6 s，合计 8.5 s |

这份样例只有 10 个块，容量太小，向量 / BM25 / RRF 三路都是满分 —— 它适合当**回归基线**（改了切分或检索，跑一遍看有没有退化），不能用来证明哪条策略更好。要比较策略，得用几十页、章节长短不一的真实文档。

## 数据边界

- 仓库只带一份虚构的示例文档（`data/samples/demo-policy.pdf`，由 `tools/make_demo_doc.py` 生成），不含任何真实合同、制度或第三方材料。
- `data/uploads/`、`data/rag.sqlite`、`eval/report.md` 都在 `.gitignore` 中：上传的文件、查询日志与评测输出只留在本机。
- 对外分发前清空 `data/` 目录即可，服务启动会自动重建空库。

## 已知限制

- **中文提问 + 英文文档时 BM25 基本失效**：两路没有共同 token，RRF 实际只剩向量一路。跨语言检索靠 bge-m3 与精排模型。
- 扫描页 OCR 每页 15–30 秒，受 `OCR_MAX_PAGES` 与 `ENHANCE_MAX_SECONDS` 双重预算限制；超预算的页会标注，不会静默失败。
- 索引常驻内存，单机单进程；百万级块需要换 ANN 索引与独立向量库。
- 增强识别后若文档已建过索引，会自动重建；文档本身变更（重新上传同名不同内容）需手动重建。

## 后续版本

| 版本 | 内容 |
|---|---|
| v1.1 | 表格跨页合并、表头继承；检索结果按章节去重 |
| v1.2 | HyDE 与多查询扩展；领域词典注入 BM25 |
| v1.3 | 增量索引（只重算变更文档）、定时清理 |
| v2.0 | 多租户与权限、RAGAS 自动评测、成本与延迟看板 |
