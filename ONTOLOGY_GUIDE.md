# OASE-KGC 本体抽取与更新流程

入口：`extract_ontology.py`。Python 3.10 及以上；只使用标准库和项目已有的 `src/kg_loader.py`，无需安装整个训练环境，也不需要 GPU。

## 1. 与论文的对应关系

依据用户提供的 `AOSE_KGC_WWW2027 (5).pdf`：第 3 节定义实体多类型、子类、互斥及关系 domain/range；第 4.2 节规定多项 schema 要求按合取解释，缺失信息归为 unknown；第 5.1 节要求仅由训练 KG 构建并缓存本体；附录 A.1 强调不能仅凭待评估关系的要求反推出实体成员资格。

本文档及实现是依据这些定义新增的可运行构建流程。论文未给出完整抽取提示词，因此代码中的提示词和置信度阈值是工程设计，不代表复现论文表 1 的确切统计值。

流程为：

1. 读取实体、关系映射和指定的三元组文件，检查 ID、行格式、头部数量，并去重。
2. 选择随机 200 条或全部三元组，保存所选 ID 三元组以复查。
3. 以实体为单位，给 LLM 提供名称、标识符和所选图内邻域，输出多类型、理由和自评置信度。默认不把原有类型或 domain/range 作为提示，避免继承来源不明的标注。
4. 清理大小写及空白，应用显式同义词映射；低置信度类型保留在 `proposed_types`，正式 `types` 为空，表示 unknown。
5. 遍历所有选中三元组，统计头尾类型分布及 `(头类型, 关系, 尾类型)` 支持数。多类型实体对每种组合各贡献一次支持；同一三元组重复出现不增加支持。
6. 逐关系向 LLM 提供完整类型计数，提议 domain/range、子类与明确互斥公理。只允许引用该关系端点已抽取的类，不凭空引入类名。
7. 校验响应 ID、类别、置信度和格式；拒绝子类环，以及与共同子类或实体多类型赋值冲突的互斥公理。
8. 输出完整本体、分项结果、变化记录和缓存。再次运行时重新统计当前图谱，复用内容相同的请求结果。

## 2. 配置接口

支持 **OpenAI-compatible Chat Completions** 协议。接口必须接受 `messages`、`model`，并返回 `choices[0].message.content`。不直接支持厂商专有的 Messages/Generate 等其他协议。

在项目目录的 PowerShell 中执行：

```powershell
Copy-Item ontology_config.example.json ontology_config.local.json
```

编辑 `ontology_config.local.json`：

- `base_url`：服务商提供的 API 基础地址，例如 `https://你的服务地址/v1`。也可填写完整 `/chat/completions` 地址。
- `model`：服务商提供的实际模型名称。
- `api_key_env`：默认 `LLM_API_KEY`，表示从该环境变量读取密钥。
- `batch_size`：每次实体抽取的实体数，默认 12。
- `context_per_entity`：每个实体最多提供 12 条邻域边，按稳定 ID 顺序取前若干条。
- `max_tokens`：单次输出上限，默认 4096；输出过长时调大，或减小批大小。
- `timeout`、`retries`：请求超时秒数及额外重试次数。
- `min_confidence`：默认 0.7。LLM 自评不是校准后的概率，需要人工抽检。
- `json_mode`：服务支持 JSON 输出模式时可设为 `true`；默认关闭以提高兼容性，仍然严格解析、校验返回 JSON。
- `type_aliases`：人工确认的同义类映射，例如 `{"Human":"Person","TV Series":"Television Series"}`。程序不自动把语义不明的类合并。

设置密钥（替换示例值）：

```powershell
$env:LLM_API_KEY = '你的API密钥'
```

也可通过 `LLM_BASE_URL` 和 `LLM_MODEL` 环境变量覆盖配置文件中的地址与模型。不要把密钥写入配置文件；程序不从配置文件读取密钥。

## 3. 两种运行模式

先检查输入和预计调用量，不发送 API 请求：

```powershell
python extract_ontology.py --config ontology_config.local.json --mode sample --dry-run
```

随机抽取 **200 条不同三元组** 的本体：

```powershell
python extract_ontology.py --config ontology_config.local.json --data-path dataset/FB15k-237 --mode sample --seed 2026
```

抽取 **全部训练三元组** 的本体：

```powershell
python extract_ontology.py --config ontology_config.local.json --data-path dataset/FB15k-237 --mode all
```

WN18RR 只需要换数据目录：

```powershell
python extract_ontology.py --config ontology_config.local.json --data-path dataset/WN18RR --mode all
```

默认输出分别位于 `import/ontology/数据集名/sample/` 和 `import/ontology/数据集名/all/`，不会覆盖源数据。图不足 200 条时使用全部数据。随机种子相同且输入图相同时，选中的三元组相同；LLM 跨次生成本身不保证完全确定，成功结果通过缓存固定。

随机模式只用这 200 条边构建上下文、类型和关系统计，不额外读取未选中的边作为 LLM 上下文。全量模式将所有选中边用于统计，但每个实体的 LLM 邻域仍受 `context_per_entity` 限制。

全量表示覆盖所选图谱中的所有边及其端点，不包括没有任何选中边的孤立实体，也不意味着穷举所有可能的类间公理。子类/互斥提议针对各关系端点出现的类型集合，不执行全局类对的平方级搜索。

## 4. 输入格式与划分

当前适配项目已有格式：

```text
数据目录/
  entity.json
  relation.json
  train2id.txt
  valid2id.txt     # 仅显式选用 valid 时必需
  test2id.txt      # 仅显式选用 test 时必需
```

实体文件支持现有 `value/id`、`label` 等字段，关系文件支持 `id`、`label`。三元组默认是 OpenKE 的 **head tail relation**，第一行可为总条数。若文件是 **head relation tail**，指定 `--triple-order hrt`。输出 `selected_triples.json` 始终是 **[head, relation, tail]**。

程序不猜测三元组顺序；错误 ID、格式、数量会明确报错。此版本不直接导入原始文本 ID 的 `train.txt`、CSV、RDF/Turtle；这类数据须先映射为上述整数 ID 格式。

论文实验保持默认 `--splits train`。如果是独立业务图谱汇总，确实需要把所有划分一起使用，可以执行：

```powershell
python extract_ontology.py --config ontology_config.local.json --mode all --splits train,valid,test --output import/ontology/FB15k-237/business-all
```

此结果不能再用于宣称“仅由训练图谱构建”的论文评估。原有实体名称、描述也应确认未嵌入测试答案；程序无法审计标注来源。只有确认旧类型来源可靠时才加 `--use-existing-types`，将其作为可被 LLM 修订的提示。

## 5. 结果说明

| 文件 | 内容 |
|---|---|
| `ontology.json` | 权威完整快照：实体、类型、关系 schema、类型级边、公理、配置及统计 |
| `entity_types.json` | 实体 ID → `types`、`proposed_types`、理由、置信度 |
| `relation_schema.json` | 关系 ID → domain/range、原始声明、端点类型计数、覆盖数、示例边、LLM 提议 |
| `class_axioms.json` | 通过一致性检查的公理及被拒绝的公理与原因 |
| `selected_triples.json` | 本次实际使用的去重三元组 |
| `changes.json` | 相对上一快照新增、删除、变更的实体/关系，含变更前后内容 |
| `history/*.json` | 每次成功更新的清单和变化记录 |
| `cache.sqlite3` | 验证成功的请求响应，供中断后续跑 |
| `dry_run.json` | 仅预检查时产生的范围及预计请求量 |

`ontology.json` 中 `type_relations` 的示意结构（非实际抽取结果）：

```json
{"head_type":"Person","relation_id":0,"relation":"livesIn","tail_type":"City","support":23}
```

需要区分两类信息：

- `domain_counts` / `range_counts`：经验类型分布，统计的是有该端点类型的三元组数，不是独立实体数，也不是必需条件。
- `domain` / `range`：达到阈值的 LLM schema 提议，列表内按 **AND** 解释。例如 `domain: ["Person", "Researcher"]` 表示同时满足二者；如果不能支持统一要求，返回空列表表示 unknown。不会自动将所有观察类型合取起来。

`source_domain` / `source_range` 保留输入文件原有声明以便对照，它们不会直接覆盖新抽取结果。互斥只在 LLM 明确提出且未发现已知冲突时记录；缺失公理不代表互斥。检查通过也不等于 OWL 逻辑完备性证明。当前输出是可审计 JSON，不是 OWL/RDF 序列化。

## 6. 更新和断点续跑

图谱新增、删除或修订后，用**相同输出目录**重跑原命令：

```powershell
python extract_ontology.py --config ontology_config.local.json --mode all
```

每次根据当前图重新统计，不把计数盲目追加到旧结果，因此重复运行不会重复计数，删除边也会减少支持数。请求内容、模型、地址、提示词等相同则命中缓存；上下文或类型分布变化时重新调用。由于实体请求按批缓存，新增实体造成分批边界变化时可能多次重算。

API 失败会重试，鉴权错误等不可恢复 HTTP 错误立即退出。失败响应不会写入缓存；中断后原命令续跑即可。没有完成整个流程时，不发布新的 `ontology.json`。分项文件与总文件之间不保证跨文件事务，消费程序应以最后写入的 `ontology.json` 为准。不要同时向同一个输出目录启动两个进程。

随机样本在图谱增删后可能变化；需要增量维护完整本体时使用 `all`。样本模式和全量模式默认分目录保存，不会将样本结果误加到全量统计中。

## 7. 后续接入论文推理模块

此脚本构建本体，不改写现有 `run.py` 的推理逻辑。通过 `json.load` 读取 `ontology.json`，按实体 ID 获取 `entities[id].types`，按关系 ID 获取 `relations[id].domain/range`，用 `axioms` 构造子类图与互斥表。

原项目 `model/filter.py` 的 range 匹配不是完整的论文第 4.2 节实现，不能把多项合取 schema 直接当作“任选其一”的候选列表使用。正式接入需遵循：每个证据三元组分别检查其自身关系的头实体 domain 和尾实体 range；子类传递闭包；正向支持优先；共同可满足子类支持 overlap；未确认情况为 unknown。对 held-out 查询不能反用目标边确认实体类型。本次未修改这部分证据评分代码。

可将 `manifest.statistics` 的类数、赋值数、子类、公理和 schema 数用于整理实验记录。schema 数按每个非空 domain/range 各计一项，与论文表 1 的口径对应，但不能把 unknown 计为已构建约束。

## 8. 验证与限制

运行本地自动测试：

```powershell
python -m unittest discover -s tests -v
```

测试覆盖随机抽样、训练集隔离、输入校验、多类型统计、重复运行和删边更新、子类环/互斥冲突、响应校验、API 重试及缓存。模拟 API 测试不能验证你的服务商兼容性和真实语义质量。首次实际运行建议使用 sample 并人工检查类型、domain/range 和互斥公理后再跑 all。

默认配置下，本项目数据的预检查估算（不含重试，不等于 token/费用估算）：FB15k-237 sample 为 200 边、368 实体、89 关系，约 120 次调用；all 为 272,115 边、14,505 实体、237 关系，约 1,446 次。请求串行执行。实际费用取决于接口价格与响应长度；缓存可减少重复请求。

很大的单关系类型词表可能超出模型上下文窗口。此版本不截断这些分布，因此遇到此情况应先使用 `type_aliases` 规范类型粒度，或选择更大上下文的模型。语义同义词归并及全局类层级补全仍需审阅。
