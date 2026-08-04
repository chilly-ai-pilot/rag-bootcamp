# Iteration 0 骨架

验收标准：`query → 检索 → 生成 → 打分` 链路跑通，分数不重要。已跑通，见下方运行结果。

## 文件结构

```
corpus/           doc-1.txt ... doc-7.txt（智能家居语料，7篇，实测355~973字）
queries.json      32条分层query，含char_start/char_end ground truth
chunking.py       固定长度切块 + 语料加载。chunk_size默认200（不是500，
                   因为doc5/6/7实测字数不到500，见文件内注释）
retrieval.py      Iteration 0用retrieve_random（完全无视query，随机返回k个chunk）
                   Iteration 1把retrieve_vector实现了就能直接换上，run_eval.py不用改
generation.py     调用Claude API；没设ANTHROPIC_API_KEY时自动降级为mock，
                   保证没有key/没有网络也能把链路跑通
scoring.py        Recall@K（按字符区间重叠判断命中）+ 按category分组统计
run_eval.py       主脚本，把上面几个串起来跑一遍，输出results.json
```

## 怎么跑

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=your_key   # 不设也能跑，会用mock生成
python run_eval.py
```

常用参数：

```bash
python run_eval.py --chunk-size 150 --overlap 30 --k 3
```

## 已知情况 / 不是bug

- 现在跑出来的Recall@K是**随机基线**，数字本身没意义，只是证明链路通了。
- generation.py在没有API key时输出`[MOCK]`占位——想看真实生成效果，设置`DEEPSEEK_API_KEY`即可，代码不用改。
- `retrieve_vector`在retrieval.py里是空的（NotImplementedError），留给Iteration 1填。

## 后续迭代怎么接

- **Iteration 1**：实现`retrieval.py`里的`retrieve_vector`（bge-base-zh + ChromaDB），在run_eval.py里把`retrieve_random`换成`retrieve_vector`，其余代码不用动。
- **Iteration 2**：在`chunking.py`里加`semantic_boundary_chunks()`、`sliding_window_chunks()`，和`fixed_length_chunks`保持同样的返回结构（list of {doc_id, start, end, text}），跑三遍`run_eval.py`对比Recall@K。
- **Iteration 5**：faithfulness/relevance打分不在`scoring.py`里现在这个函数里加，另开一个模块，避免和Recall@K的逻辑混在一起。
