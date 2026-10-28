# 根据抽取本体生成 OASE-KGC 证据

入口：`evidence_pipeline.py`。直接读取 `extract_ontology.py` 生成的 **ontology.json**，无需格式转换。不要传入 cache.sqlite3 或单独的 entity_types.json。

## 1. 流程与论文对应

实现依据论文第 4.1–4.3 节、附录 A.2–A.4：

1. **训练图结构索引**：仅将 train2id.txt 的边加入索引。valid/test 仅提供查询。
2. **一跳抽取（式 2）**：关系余弦相关性与 RotatE 复数表示的三元组 L2 距离可靠性加权。
3. **多跳抽取（式 3）**：复数关系逐元素乘积表示路径关系；结合查询终点距离和长度衰减。按 beam 限制保留路径前缀，不重复访问实体。
4. **本体校准（式 4–5）**：逐边检查自身关系的 domain/range，匹配或子类为 1，严格父类为 0.9，有共同可满足子类支持的 overlap 为 0.65，明确互斥为 0.1，unknown 为 0.6。多个 schema 要求按乘积合取；路径权重是所有边头尾兼容度的乘积。
5. **预算选择（式 6–7、11、14）**：以证据质量和关系/实体/边覆盖组成目标，按单位成本边际收益贪心选择，再与最佳可行单条证据比较。覆盖采用最高质量支持，保留有增益的部分重叠路径。
6. **候选实体（附录 A.3）**：RotatE top-K 加校准后路径池中的高分终点，去重；不插入真实答案，不过滤已知答案。候选构建不改变证据池或选择目标。
7. **LLM 输入**：保存查询、证据、候选列表和要求返回候选索引排序的 `llm_messages`。本程序不调用 LLM，不需要 API 密钥。

FB15k-237 默认 L=3、W=30、一跳最多150、路径最多100；WN18RR 默认 L=2、W=20、一跳最多80、路径最多50。预算默认3000，其余参数在 `defaults()` 中与论文表6对应。

工程约定：

- 为避免读取 gold answer，屏蔽查询起点上与查询关系相同的所有直接边。这比“只排除 held-out 目标边”更保守，可能减少证据，实验报告中应说明。
- 合取兼容度使用各要求分数的乘积，是对论文多项同时要求的数值实现选择。
- 观察到的实体类型赋值用于见证类的非空性；同时具备多重祖先的已见证子类，可支持 overlap。与已知互斥冲突的共同子类不会用于支持 overlap。
- 仅新增证据构建入口；旧版 `model/extractor.py`、`run.py` 不会自动改用它。旧抽取器包含 gold-tail 奖励，不应代替此入口用于独立评估。

## 2. 服务器准备

在服务器项目根目录操作，Python 3.10+。使用导出的 numpy 权重时只需：

```bash
pip install numpy
```

准备两个输入：

```text
import/ontology/FB15k-237/all/ontology.json

import/KGE_model/FB15k-237/
  entity_embedding.npy
  relation_embedding.npy
  config.json
```

也可把 `--embeddings` 指向 `checkpoint.pt`，但这需要安装 PyTorch 并支持 `torch.load(weights_only=True)`。权重必须是项目 `model/KGE_model.py` 的 RotatE 格式：实体为 `[N,2d]`，关系为 `[R,d]` 的相位参数。其他库权重需转换，不可直接假定相同格式。

程序不会自动训练、生成随机权重或调用外部模型。没有权重时，可用现有训练模块显式训练。例如 FB15k-237：

```bash
python3 model/KGE_model.py --mode train \
  --data_path dataset/FB15k-237 --dataset_name FB15k-237 \
  --hidden_dim 200 --gamma 9 --learning_rate 0.00005 --max_steps 100000
```

该训练命令可能耗时较长；已有训练权重时直接复用。使用者需确认权重仅基于训练边训练，以及实体/关系 ID 与当前数据完全一致；形状校验无法证明训练来源。

## 3. 先抽取少量查询

你正在生成 sample 本体，完成后可先用它验证流程：

```bash
python3 evidence_pipeline.py \
  --data-path dataset/FB15k-237 \
  --ontology import/ontology/FB15k-237/sample/ontology.json \
  --embeddings import/KGE_model/FB15k-237 \
  --split test --prediction tail --limit 10 \
  --output output/evidence/FB15k-237/sample_10.jsonl
```

注意两个范围不同：本体 sample 是由200条训练边构建；`--limit 10` 是处理10个预测查询。证据始终从整个训练图检索。本体未覆盖的实体、关系使用 unknown 分数，而不是继承原文件来源不明的类型。因此 sample 用于调试，正式实验建议使用 all 本体。

## 4. Token 预算与正式运行

不指定 tokenizer 时，以 UTF-8 字节数作为 byte-level BPE 的保守上界。此时预算3000不等于精确3000 tokens，通常会选出较少证据。

对论文中的 o200k_base 计数：

```bash
pip install tiktoken

python3 evidence_pipeline.py \
  --data-path dataset/FB15k-237 \
  --ontology import/ontology/FB15k-237/all/ontology.json \
  --embeddings import/KGE_model/FB15k-237 \
  --encoding o200k_base --token-budget 3000 \
  --split test --prediction both \
  --output output/evidence/FB15k-237/test_both.jsonl
```

`tiktoken` 初次加载编码可能需要下载词表。若实际使用其他 LLM，应使用与该模型对应的本地 tokenizer：

```bash
python3 evidence_pipeline.py \
  --data-path dataset/FB15k-237 \
  --ontology import/ontology/FB15k-237/all/ontology.json \
  --embeddings import/KGE_model/FB15k-237 \
  --tokenizer /你的服务器路径/tokenizer \
  --split test --prediction tail --limit 10 \
  --output output/evidence/FB15k-237/local_tokenizer_10.jsonl
```

本地 tokenizer 模式需要 `transformers`，不启用远程代码。`--tokenizer` 和 `--encoding` 二选一。

预算只约束证据；问题、候选、系统提示和返回结果另外占用上下文。可加 `--context-window 32768 --response-reserve 1024` 做整段输入的估算检查。程序计数消息正文并加32开销，厂商聊天模板可能不同，因此发送前仍需按照实际 API 模板确认总长度；这不是服务端精确 token 计费结果。

`--prediction both` 对每行查询先输出 tail，再输出 head。头预测将训练边反向、关系向量取复共轭，并交换 domain/range；文本明确标记 inverse of。`--limit` 计算输出查询条数，因此 both 模式 limit=10 通常是5行测试三元组的双向预测。

## 5. 结果及中间进度

每处理完一个查询，就把完整 JSON 写入输出 JSONL 并刷新文件，不必等全量结束。另开终端可查看第一条：

```bash
head -n 1 output/evidence/FB15k-237/sample_10.jsonl | python3 -m json.tool
```

主要字段：

| 字段 | 含义 |
|---|---|
| `query_index` | 原输入文件中0起始的查询行索引（不含数量头部） |
| `prediction` | tail 或 head |
| `query_entity_id` | 已知端实体ID，头预测时为原尾实体 |
| `evidence_text` | 可直接放进 prompt 的证据文本 |
| `selected_evidence` | 选中证据、三元组ID、结构分数、本体校准、成本 |
| `evidence_pool` | 预算选择前的全部候选证据，用于分析/消融 |
| `candidates` | 候选索引、实体ID、名称、类型、候选来源 |
| `llm_messages` | 可直接传入兼容聊天接口的 messages 数组 |
| `evidence_cost` | 最终拼接证据的实际计数成本 |
| `provenance` | train 来源、输入哈希、本体模式、是否使用gold等 |

示意证据（不是运行结果）：

```text
- "Alice" --["worksAt"]--> "University A"
- "Alice" --["bornIn"]--> "City A" --["locatedIn"]--> "Country A"
```

程序保留原始划分行顺序及重复查询，以便与测试答案对齐；`llm_messages` 中不包含 gold answer 或 gold 标识。头预测的证据三元组 ID 按反向解释，配合 `prediction=head` 和文本中的 inverse of 使用。

输出文件已存在时程序拒绝覆盖，避免误删已有实验。中断时已刷新的行保留，本版本未实现自动断点续跑；重跑使用新的 `--output` 文件名。完整 evidence_pool 会增加文件体积。

## 6. 把结果交给 LLM

读取一条结果后直接使用其 messages：

```python
import json

with open("output/evidence/FB15k-237/sample_10.jsonl", encoding="utf-8") as f:
    record = json.loads(next(f))

messages = record["llm_messages"]
# 将 messages 传给你的 LLM 调用函数。
# 期望返回：{"ranking": [3, 0, 2, 1, ...]}
# 将候选索引映射回实体 ID：
# candidate_entity_ids = [record["candidates"][i]["entity_id"] for i in ranking]
```

评估时校验每个候选索引恰好出现一次，真实答案从原测试集按 query_index 获取，并与 prediction 对应。不要将答案加入 prompt，也不要强制放进候选列表。

本程序负责证据及候选构建，不负责 API 调用或评估指标。论文的 filtered full-entity MRR/Hits 还需要：将候选排序放在完整 RotatE 排序之前，其余实体保持原序；只在指标计算阶段过滤其他已知答案；答案未进入候选集的查询也必须纳入。仅对候选内答案算指标会高估效果。

## 7. 自定义查询

尾预测 JSONL：

```json
{"query_head_id":0,"query_relation_id":1}
```

头预测则提供 `query_tail_id`，both 时提供两端作为双向查询的已知端。使用 `--queries queries.jsonl` 替代测试划分作为查询来源；程序不会使用另一端给证据评分。

## 8. 验证范围

测试命令：`python3 -m unittest discover -s tests -v`。

新增测试覆盖当前本体格式、unknown、正向支持优先、合取、overlap、一跳/多跳公式、逐边校准、反向关系、预算、最佳单证据比较、候选排序，以及完整命令行输出和来源不匹配拒绝。已使用小型合成权重完成端到端验证；本地未发现可用的训练权重及服务器抽取后的实际本体，因此尚未进行真实数据的 LLM 指标评测。
