# Judge Options Update - DeepSeek Integration

## Changes Summary

Updated `iteration6/run_eval.py` to support DeepSeek as an alternative Judge model alongside Qwen.

## Modifications

### 1. **Argument Parser**
- Changed `--judge-mode` choices from `["llm", "ragas", "none"]` to `["qwen", "deepseek", "ragas", "none"]`
- "llm" renamed to "qwen" for clarity
- Default value: "qwen"

### 2. **Judge Client Logic**

Updated three functions to support DeepSeek:

#### `batch_evaluate_combined()`
- Added conditional logic to choose API client based on `args.judge_mode`
- If `judge_mode == "deepseek"`: uses `DEEPSEEK_API_KEY` and `https://api.deepseek.com`
- If `judge_mode == "qwen"`: uses `ALI_API_KEY` and `ALI_BASE_URL`
- Each judge mode creates a separate `AsyncOpenAI` client instance (no session mixing)

#### `batch_evaluate_faithfulness()`
- Same pattern as above for Faithfulness-only evaluation
- Supports both qwen and deepseek judge modes

#### `batch_evaluate_relevance()`
- Same pattern for Answer Relevance evaluation
- Supports both qwen and deepseek judge modes

### 3. **Display Names**
- Updated `judge_mode_names` dict: `{"qwen": "Qwen Judge", "deepseek": "DeepSeek Judge", "ragas": "Ragas Judge"}`
- Updated print statements to show correct judge name in output

### 4. **Documentation**
- Updated module docstring with DeepSeek usage examples
- Updated help text for `--judge-mode` argument

## Environment Variables Required

### For Qwen (default):
```bash
export ALI_API_KEY="your_qwen_api_key"
export ALI_BASE_URL="your_qwen_base_url"
```

### For DeepSeek:
```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```

## Usage Examples

### Use Qwen Judge (default):
```bash
python3 run_eval.py --chunking-strategy fixed_100_50 --retrieval-mode vector
# or explicitly
python3 run_eval.py --chunking-strategy fixed_100_50 --retrieval-mode vector --judge-mode qwen
```

### Use DeepSeek Judge:
```bash
python3 run_eval.py --chunking-strategy fixed_100_50 --retrieval-mode vector --judge-mode deepseek
```

### Use Ragas Judge:
```bash
python3 run_eval.py --chunking-strategy fixed_100_50 --retrieval-mode vector --judge-mode ragas
```

### Disable Judge:
```bash
python3 run_eval.py --chunking-strategy fixed_100_50 --retrieval-mode vector --judge-mode none
```

## Key Design Decisions

1. **Separate Sessions**: Each judge mode creates its own `AsyncOpenAI` client instance, ensuring no context mixing between generation and evaluation
2. **Model Selection**: Using `deepseek-chat` (not reasoner) for judge evaluation
3. **Backward Compatibility**: Existing scripts using default judge will use "qwen" automatically
4. **Fallback on Error**: If Qwen (Aliyun) has account issues, users can now switch to DeepSeek easily

## Testing

Verified with:
```bash
python3 run_eval.py --help
python3 -m py_compile run_eval.py  # Syntax check
```

Both commands completed successfully.
