"""
Iteration 8: HTML 报告生成器（带自愈审核 UI）

功能：
- 读取 data/ 目录下所有历史评估结果
- 生成交互式 HTML 报告（使用 Chart.js）
- 支持自愈审核：显示待审核文件，允许批量通过
- 支持 GitHub Pages 直接访问

使用方法：
    # 从 data/ 目录读取结果，输出到 docs/
    python generate_report.py --data-dir ../data --output-dir ../docs
    
    # 使用默认路径
    python generate_report.py
"""

import os
import json
import glob
import argparse
from datetime import datetime
from typing import List, Dict


def load_pending_reviews(review_dir: str) -> List[Dict]:
    """加载所有待审核文件
    
    参数:
        review_dir: 审核文件目录
    
    返回:
        待审核文件列表
    """
    if not os.path.exists(review_dir):
        return []
    
    pending_reviews = []
    
    for filename in os.listdir(review_dir):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(review_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get('status') == 'pending':
                data['filename'] = filename
                pending_reviews.append(data)
        except Exception as e:
            print(f"⚠️  Failed to load review file {filename}: {e}")
            continue
    
    # 按创建时间倒序排序
    pending_reviews.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return pending_reviews


def load_all_results(data_dir: str) -> List[Dict]:
    """加载所有历史评估结果
    
    参数:
        data_dir: 结果文件目录
    
    返回:
        按时间排序的结果列表
    """
    pattern = os.path.join(data_dir, "results_*.json")
    result_files = glob.glob(pattern)
    
    if not result_files:
        print(f"⚠️  No result files found in {data_dir}")
        return []
    
    results = []
    for file_path in result_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 确保有 metadata
                if 'metadata' not in data:
                    # 尝试从文件名提取时间戳
                    filename = os.path.basename(file_path)
                    parts = filename.replace('.json', '').split('_')
                    if len(parts) >= 5:  # results_strategy_mode_date_time.json
                        timestamp = f"{parts[-2]}_{parts[-1]}"
                    else:
                        timestamp = "unknown"
                    
                    data['metadata'] = {
                        'timestamp': timestamp,
                        'evaluation_date': 'Unknown',
                        'source_file': filename
                    }
                
                data['_source_file'] = os.path.basename(file_path)
                results.append(data)
        except Exception as e:
            print(f"⚠️  Failed to load {file_path}: {e}")
    
    # 按时间戳排序
    results.sort(key=lambda x: x['metadata'].get('timestamp', ''), reverse=False)
    
    print(f"✅ Loaded {len(results)} evaluation results from {data_dir}")
    return results


def extract_metrics(results: List[Dict]) -> Dict:
    """从结果中提取关键指标
    
    返回:
        {
            'timestamps': [...],
            'hit_rates': [...],
            'faithfulness_scores': [...],
            'relevance_scores': [...],
            'rejection_rates': [...],
            'mrr_scores': [...]
        }
    """
    metrics = {
        'timestamps': [],
        'dates': [],
        'hit_rates': [],
        'faithfulness_scores': [],
        'relevance_scores': [],
        'rejection_rates': [],
        'mrr_scores': [],
        'configs': []
    }
    
    for result in results:
        metadata = result.get('metadata', {})
        scores = result.get('scores', {})
        faithfulness_analysis = result.get('faithfulness_analysis', {})
        relevance_analysis = result.get('relevance_analysis', {})
        mrr_scores = result.get('mrr_scores', {})
        
        # 时间戳
        timestamp = metadata.get('timestamp', 'unknown')
        metrics['timestamps'].append(timestamp)
        
        # 可读日期
        eval_date = metadata.get('evaluation_date', '')
        if eval_date:
            try:
                dt = datetime.fromisoformat(eval_date)
                readable_date = dt.strftime('%Y-%m-%d %H:%M')
            except:
                readable_date = timestamp
        else:
            readable_date = timestamp
        metrics['dates'].append(readable_date)
        
        # Hit Rate（总体）
        # scores 可能是 float（旧格式）或 dict（新格式）
        if isinstance(scores, dict) and 'overall' in scores:
            # 新格式：{"overall": 0.85, "category1": 0.9, ...}
            hit_rate = scores.get('overall', 0)
        else:
            # 旧格式：{"factual": {"hit": 10, "count": 12}, ...}
            if scores and isinstance(list(scores.values())[0], dict):
                total_hit = sum(v.get('hit', 0) for v in scores.values() if isinstance(v, dict))
                total_queries = sum(v.get('count', 0) for v in scores.values() if isinstance(v, dict))
                hit_rate = total_hit / total_queries if total_queries > 0 else 0
            else:
                # 简单格式：直接取 overall 或默认 0
                hit_rate = scores.get('overall', 0) if isinstance(scores, dict) else 0
        metrics['hit_rates'].append(hit_rate)
        
        # Faithfulness Score（平均）
        # 优先从 faithfulness_analysis 读取，否则从 results 计算
        if faithfulness_analysis and 'overall_avg_faithfulness' in faithfulness_analysis:
            faith_score = faithfulness_analysis.get('overall_avg_faithfulness', 0)
        else:
            # 从 results 中计算
            results_list = result.get('results', [])
            faith_scores = [r.get('faithfulness_score') for r in results_list if r.get('faithfulness_score') is not None]
            faith_score = sum(faith_scores) / len(faith_scores) if faith_scores else 0
        metrics['faithfulness_scores'].append(faith_score)
        
        # Relevance Score（平均）
        # 优先从 relevance_analysis 读取，否则从 results 计算
        if relevance_analysis and 'overall_avg_relevance' in relevance_analysis:
            rel_score = relevance_analysis.get('overall_avg_relevance', 0)
        else:
            # 从 results 中计算
            results_list = result.get('results', [])
            rel_scores = [r.get('relevance_score') for r in results_list if r.get('relevance_score') is not None]
            rel_score = sum(rel_scores) / len(rel_scores) if rel_scores else 0
        metrics['relevance_scores'].append(rel_score)
        
        # Rejection Rate
        results_list = result.get('results', [])
        if results_list:
            rejected_count = sum(1 for r in results_list if r.get('rejected', False))
            rejection_rate = rejected_count / len(results_list)
        else:
            rejection_rate = 0
        metrics['rejection_rates'].append(rejection_rate)
        
        # MRR Score（总体）
        # mrr_scores 可能是 float 或 dict
        if isinstance(mrr_scores, dict):
            mrr = mrr_scores.get('overall', 0)
        elif isinstance(mrr_scores, (int, float)):
            mrr = mrr_scores
        else:
            mrr = 0
        metrics['mrr_scores'].append(mrr)
        
        # 配置信息
        config = metadata.get('model_config', result.get('config', {}))
        config_str = f"{config.get('chunking_strategy', 'N/A')}+{config.get('retrieval_mode', 'N/A')}+{config.get('rerank_mode', 'N/A')}"
        metrics['configs'].append(config_str)
    
    return metrics


def generate_html_report(results: List[Dict], pending_reviews: List[Dict], output_path: str):
    """生成 HTML 报告
    
    参数:
        results: 所有评估结果
        pending_reviews: 待审核文件列表
        output_path: 输出文件路径
    """
    if not results:
        print("⚠️  No results to generate report")
        return
    
    # 提取指标
    metrics = extract_metrics(results)
    latest_result = results[-1]
    
    # 计算对比（最新 vs 上一次）
    comparison = None
    if len(results) >= 2:
        prev_result = results[-2]
        comparison = {
            'hit_rate_diff': metrics['hit_rates'][-1] - metrics['hit_rates'][-2],
            'faithfulness_diff': metrics['faithfulness_scores'][-1] - metrics['faithfulness_scores'][-2],
            'relevance_diff': metrics['relevance_scores'][-1] - metrics['relevance_scores'][-2],
            'rejection_rate_diff': metrics['rejection_rates'][-1] - metrics['rejection_rates'][-2],
            'mrr_diff': metrics['mrr_scores'][-1] - metrics['mrr_scores'][-2]
        }
    
    # 生成 HTML
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG System Evaluation Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f7fa;
            color: #2c3e50;
            padding: 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            border-bottom: 2px solid #dee2e6;
        }}
        
        .tab {{
            padding: 12px 24px;
            background: white;
            border: none;
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            color: #6c757d;
            transition: all 0.3s;
        }}
        
        .tab:hover {{
            background: #f8f9fa;
            color: #495057;
        }}
        
        .tab.active {{
            background: #667eea;
            color: white;
            border-bottom: 2px solid #667eea;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        .card h3 {{
            font-size: 0.9em;
            color: #7f8c8d;
            text-transform: uppercase;
            margin-bottom: 10px;
            letter-spacing: 0.5px;
        }}
        
        .card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .card .diff {{
            font-size: 0.9em;
            padding: 4px 8px;
            border-radius: 4px;
            display: inline-block;
        }}
        
        .diff.positive {{
            background: #d4edda;
            color: #155724;
        }}
        
        .diff.negative {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .diff.neutral {{
            background: #e2e3e5;
            color: #383d41;
        }}
        
        .chart-container {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .chart-container h2 {{
            margin-bottom: 20px;
            color: #2c3e50;
            font-size: 1.5em;
        }}
        
        .chart-wrapper {{
            position: relative;
            height: 400px;
        }}
        
        .details-table {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        
        .details-table h2 {{
            margin-bottom: 20px;
            color: #2c3e50;
            font-size: 1.5em;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        
        th {{
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #495057;
            border-bottom: 2px solid #dee2e6;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .badge {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            display: inline-block;
        }}
        
        .badge.success {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge.warning {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .badge.danger {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .metadata {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .metadata h3 {{
            margin-bottom: 15px;
            color: #2c3e50;
        }}
        
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .metadata-item {{
            padding: 10px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        
        .metadata-item .label {{
            font-size: 0.85em;
            color: #6c757d;
            margin-bottom: 4px;
        }}
        
        .metadata-item .value {{
            font-weight: 600;
            color: #2c3e50;
        }}
        
        .review-container {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        .review-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        
        .review-header h2 {{
            color: #2c3e50;
            font-size: 1.5em;
        }}
        
        .review-actions {{
            display: flex;
            gap: 10px;
        }}
        
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.95em;
            font-weight: 600;
            transition: all 0.3s;
        }}
        
        .btn-primary {{
            background: #667eea;
            color: white;
        }}
        
        .btn-primary:hover {{
            background: #5568d3;
        }}
        
        .btn-primary:disabled {{
            background: #c0c5dd;
            cursor: not-allowed;
        }}
        
        .btn-secondary {{
            background: #6c757d;
            color: white;
        }}
        
        .btn-secondary:hover {{
            background: #5a6268;
        }}
        
        .review-list {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        
        .review-item {{
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            transition: all 0.3s;
        }}
        
        .review-item:hover {{
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .review-item.selected {{
            border-color: #667eea;
            background: #f8f9ff;
        }}
        
        .review-checkbox {{
            width: 20px;
            height: 20px;
            cursor: pointer;
        }}
        
        .review-item-header {{
            display: flex;
            align-items: start;
            gap: 15px;
            margin-bottom: 15px;
        }}
        
        .review-item-content {{
            flex: 1;
        }}
        
        .review-query {{
            font-size: 1.1em;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 8px;
        }}
        
        .review-meta {{
            display: flex;
            gap: 15px;
            font-size: 0.9em;
            color: #6c757d;
            margin-bottom: 10px;
        }}
        
        .review-reason {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        
        .reason-retrieval-miss {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .reason-low-rank {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .reason-layer1-rejection {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        
        .reason-low-score-rejection {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        
        .review-answer {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
            margin-top: 10px;
        }}
        
        .review-answer-label {{
            font-size: 0.85em;
            color: #6c757d;
            margin-bottom: 5px;
            font-weight: 600;
        }}
        
        .review-answer-text {{
            color: #2c3e50;
            line-height: 1.6;
        }}
        
        .editable {{
            min-height: 40px;
            padding: 8px;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            background: white;
            transition: all 0.3s;
        }}
        
        .editable:hover {{
            border-color: #667eea;
        }}
        
        .editable:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}
        
        .edit-hint {{
            font-size: 0.8em;
            color: #6c757d;
            margin-top: 5px;
            font-style: italic;
        }}
        
        .empty-state {{
            text-align: center;
            padding: 60px 20px;
            color: #6c757d;
        }}
        
        .empty-state svg {{
            width: 80px;
            height: 80px;
            margin-bottom: 20px;
            opacity: 0.5;
        }}
        
        .empty-state h3 {{
            font-size: 1.3em;
            margin-bottom: 10px;
        }}
        
        footer {{
            text-align: center;
            padding: 30px;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .summary-cards {{
                grid-template-columns: 1fr;
            }}
            
            header h1 {{
                font-size: 1.8em;
            }}
            
            .chart-wrapper {{
                height: 300px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 RAG System Evaluation Report</h1>
            <p>Continuous Evaluation Dashboard - Iteration 8 (Self-Healing)</p>
            <p style="font-size: 0.95em; margin-top: 10px;">Latest: {metrics['dates'][-1]} | Total Runs: {len(results)} | Pending Reviews: {len(pending_reviews)}</p>
        </header>
        
        <div class="tabs">
            <button class="tab active" onclick="switchTab('metrics')">📊 Metrics</button>
            <button class="tab" onclick="switchTab('review')">🔍 Review ({len(pending_reviews)})</button>
        </div>
        
        <div id="metrics-tab" class="tab-content active">
"""

    # Summary Cards
    latest_metadata = latest_result.get('metadata', {})
    html_content += f"""
        <div class="summary-cards">
            <div class="card">
                <h3>Hit Rate</h3>
                <div class="value">{metrics['hit_rates'][-1]:.1%}</div>
"""
    if comparison:
        diff_class = 'positive' if comparison['hit_rate_diff'] >= 0 else 'negative'
        diff_symbol = '+' if comparison['hit_rate_diff'] >= 0 else ''
        html_content += f"""<div class="diff {diff_class}">{diff_symbol}{comparison['hit_rate_diff']:.1%} vs previous</div>"""
    html_content += """
            </div>
            
            <div class="card">
                <h3>Faithfulness</h3>
"""
    html_content += f"""<div class="value">{metrics['faithfulness_scores'][-1]:.3f}</div>"""
    if comparison:
        diff_class = 'positive' if comparison['faithfulness_diff'] >= 0 else 'negative'
        diff_symbol = '+' if comparison['faithfulness_diff'] >= 0 else ''
        html_content += f"""<div class="diff {diff_class}">{diff_symbol}{comparison['faithfulness_diff']:.3f} vs previous</div>"""
    html_content += """
            </div>
            
            <div class="card">
                <h3>Relevance</h3>
"""
    html_content += f"""<div class="value">{metrics['relevance_scores'][-1]:.3f}</div>"""
    if comparison:
        diff_class = 'positive' if comparison['relevance_diff'] >= 0 else 'negative'
        diff_symbol = '+' if comparison['relevance_diff'] >= 0 else ''
        html_content += f"""<div class="diff {diff_class}">{diff_symbol}{comparison['relevance_diff']:.3f} vs previous</div>"""
    html_content += """
            </div>
            
            <div class="card">
                <h3>Rejection Rate</h3>
"""
    html_content += f"""<div class="value">{metrics['rejection_rates'][-1]:.1%}</div>"""
    if comparison:
        diff_class = 'negative' if comparison['rejection_rate_diff'] >= 0 else 'positive'  # 拒答率低更好
        diff_symbol = '+' if comparison['rejection_rate_diff'] >= 0 else ''
        html_content += f"""<div class="diff {diff_class}">{diff_symbol}{comparison['rejection_rate_diff']:.1%} vs previous</div>"""
    html_content += """
            </div>
            
            <div class="card">
                <h3>MRR Score</h3>
"""
    html_content += f"""<div class="value">{metrics['mrr_scores'][-1]:.3f}</div>"""
    if comparison:
        diff_class = 'positive' if comparison['mrr_diff'] >= 0 else 'negative'
        diff_symbol = '+' if comparison['mrr_diff'] >= 0 else ''
        html_content += f"""<div class="diff {diff_class}">{diff_symbol}{comparison['mrr_diff']:.3f} vs previous</div>"""
    html_content += """
            </div>
        </div>
"""

    # Metadata
    corpus_stats = latest_metadata.get('corpus_stats', {})
    model_config = latest_metadata.get('model_config', {})
    html_content += f"""
        <div class="metadata">
            <h3>📋 Configuration & Corpus Info</h3>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <div class="label">Documents</div>
                    <div class="value">{corpus_stats.get('num_documents', 'N/A')}</div>
                </div>
                <div class="metadata-item">
                    <div class="label">Queries</div>
                    <div class="value">{corpus_stats.get('num_queries', 'N/A')}</div>
                </div>
                <div class="metadata-item">
                    <div class="label">Chunking Strategy</div>
                    <div class="value">{model_config.get('chunking_strategy', 'N/A')}</div>
                </div>
                <div class="metadata-item">
                    <div class="label">Retrieval Mode</div>
                    <div class="value">{model_config.get('retrieval_mode', 'N/A')}</div>
                </div>
                <div class="metadata-item">
                    <div class="label">Rerank Mode</div>
                    <div class="value">{model_config.get('rerank_mode', 'N/A')}</div>
                </div>
                <div class="metadata-item">
                    <div class="label">Judge Mode</div>
                    <div class="value">{model_config.get('judge_mode', 'N/A')}</div>
                </div>
                <div class="metadata-item">
                    <div class="label">Rejection Enabled</div>
                    <div class="value">{'Yes' if model_config.get('rejection_enabled', False) else 'No'}</div>
                </div>
                <div class="metadata-item">
                    <div class="label">Rejection Preset</div>
                    <div class="value">{model_config.get('rejection_preset', 'custom')}</div>
                </div>
            </div>
        </div>
"""

    # Charts
    html_content += """
        <div class="chart-container">
            <h2>📈 Metrics Trend Over Time</h2>
            <div class="chart-wrapper">
                <canvas id="metricsChart"></canvas>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>📊 Hit Rate & Rejection Rate</h2>
            <div class="chart-wrapper">
                <canvas id="ratesChart"></canvas>
            </div>
        </div>
"""

    # History Table
    html_content += """
        <div class="details-table">
            <h2>📜 Evaluation History</h2>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Config</th>
                        <th>Hit Rate</th>
                        <th>Faithfulness</th>
                        <th>Relevance</th>
                        <th>Rejection Rate</th>
                        <th>MRR</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    for i in range(len(results)):
        config_str = metrics['configs'][i]
        hit_rate = metrics['hit_rates'][i]
        faith = metrics['faithfulness_scores'][i]
        rel = metrics['relevance_scores'][i]
        rej_rate = metrics['rejection_rates'][i]
        mrr = metrics['mrr_scores'][i]
        
        # Badge for hit rate
        if hit_rate >= 0.8:
            hit_badge = 'success'
        elif hit_rate >= 0.6:
            hit_badge = 'warning'
        else:
            hit_badge = 'danger'
        
        html_content += f"""
                    <tr>
                        <td>{metrics['dates'][i]}</td>
                        <td><code>{config_str}</code></td>
                        <td><span class="badge {hit_badge}">{hit_rate:.1%}</span></td>
                        <td>{faith:.3f}</td>
                        <td>{rel:.3f}</td>
                        <td>{rej_rate:.1%}</td>
                        <td>{mrr:.3f}</td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
        </div>
        
        </div>
        <!-- End Metrics Tab -->
        
        <!-- Review Tab -->
        <div id="review-tab" class="tab-content">
            <div class="review-container">
                <div class="review-header">
                    <h2>🔍 Self-Healing Review</h2>
                    <div class="review-actions">
                        <button class="btn btn-primary" id="approveBtn" onclick="approveSelected()">🚀 Approve via GitHub Actions</button>
                    </div>
                </div>
                <div style="background: #e7f3ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #2196F3;">
                    <strong>💡 How to approve reviews:</strong>
                    <ol style="margin: 10px 0 0 20px; line-height: 1.8;">
                        <li>Review the Q&A pairs below (you can edit them inline if needed)</li>
                        <li>Click the "🚀 Approve via GitHub Actions" button above</li>
                        <li>On the GitHub Actions page, click "Run workflow" → Choose "yes" → Click the green "Run workflow" button</li>
                        <li>Wait for the workflow to complete - it will automatically approve all reviews and add them to the corpus</li>
                    </ol>
                    <p style="margin-top: 10px; font-size: 0.9em; color: #666;">
                        Note: Currently, inline edits are for review only. To modify Q&A pairs before approval, edit the JSON files in <code>iteration8/review/</code> directly.
                    </p>
                </div>
"""
    
    if pending_reviews:
        html_content += """
                <div class="review-list">
"""
        for review in pending_reviews:
            query = review.get('query', '')
            ground_truth = review.get('ground_truth', '')
            trigger_reason = review.get('trigger_reason', '')
            rejection_reason = review.get('rejection_reason', '')
            created_at = review.get('created_at', '')
            filename = review.get('filename', '')
            doc_id = review.get('source', {}).get('doc_id', '')
            
            # Format created_at
            try:
                dt = datetime.fromisoformat(created_at)
                formatted_date = dt.strftime('%Y-%m-%d %H:%M')
            except:
                formatted_date = created_at
            
            # Determine reason class
            reason_class = 'reason-retrieval-miss'
            if 'low_rank' in trigger_reason:
                reason_class = 'reason-low-rank'
            elif 'low_score' in trigger_reason.lower():
                reason_class = 'reason-low-score-rejection'
            
            html_content += f"""
                    <div class="review-item" data-filename="{filename}">
                        <div class="review-item-content">
                            <div class="review-query">
                                <strong>Question:</strong>
                                <div contenteditable="true" class="editable" data-field="query" style="margin-top: 5px;">{query}</div>
                                <div class="edit-hint">Click to edit query</div>
                            </div>
                            <div class="review-meta">
                                <span>📅 {formatted_date}</span>
                                <span>📄 {doc_id or 'N/A'}</span>
                                <span class="review-reason {reason_class}">{trigger_reason}</span>
                            </div>
"""
            if rejection_reason:
                html_content += f"""
                                <div style="font-size: 0.9em; color: #6c757d; margin-top: 5px;">
                                    Rejection: {rejection_reason}
                                </div>
"""
            html_content += f"""
                                <div class="review-answer">
                                    <div class="review-answer-label">Ground Truth Answer:</div>
                                    <div contenteditable="true" class="review-answer-text editable" data-field="ground_truth">{ground_truth}</div>
                                    <div class="edit-hint">Click to edit answer</div>
                                </div>
                        </div>
                    </div>
"""
        html_content += """
                </div>
"""
    else:
        html_content += """
                <div class="empty-state">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <h3>All Clear!</h3>
                    <p>No pending reviews. The system is performing well.</p>
                </div>
"""
    
    html_content += """
            </div>
        </div>
        <!-- End Review Tab -->
"""

    # Footer
    html_content += f"""
        <footer>
            <p>Generated by RAG Bootcamp Iteration 8 - Self-Healing RAG System</p>
            <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
    
    <script>
        // Tab Switching
        function switchTab(tabName) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.remove('active');
            }});
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            
            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
        }}
        
        function approveSelected() {{
            // 直接跳转到 GitHub Actions
            const repoUrl = 'https://github.com/' + getGitHubRepo();
            const workflowUrl = repoUrl + '/actions/workflows/approve_reviews.yml';
            
            if (confirm('Ready to approve reviews?\\n\\nThis will redirect you to GitHub Actions to run the approval workflow.\\n\\nClick OK to continue.')) {{
                window.open(workflowUrl, '_blank');
            }}
        }}
        
        function getGitHubRepo() {{
            // 尝试从当前 URL 推断仓库路径
            const hostname = window.location.hostname;
            if (hostname.includes('github.io')) {{
                // GitHub Pages: username.github.io/repo-name
                const pathParts = window.location.pathname.split('/').filter(p => p);
                if (pathParts.length > 0) {{
                    const username = hostname.split('.')[0];
                    const repo = pathParts[0];
                    return username + '/' + repo;
                }}
            }}
            // 默认值 - 用户需要在生成报告时配置
            return 'chilly-ai-pilot/rag-bootcamp';
        }}
        
        // Metrics Trend Chart
        const metricsCtx = document.getElementById('metricsChart').getContext('2d');
        const metricsChart = new Chart(metricsCtx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(metrics['dates'])},
                datasets: [
                    {{
                        label: 'Faithfulness',
                        data: {json.dumps(metrics['faithfulness_scores'])},
                        borderColor: 'rgb(102, 126, 234)',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.3,
                        fill: true
                    }},
                    {{
                        label: 'Relevance',
                        data: {json.dumps(metrics['relevance_scores'])},
                        borderColor: 'rgb(118, 75, 162)',
                        backgroundColor: 'rgba(118, 75, 162, 0.1)',
                        tension: 0.3,
                        fill: true
                    }},
                    {{
                        label: 'MRR',
                        data: {json.dumps(metrics['mrr_scores'])},
                        borderColor: 'rgb(255, 159, 64)',
                        backgroundColor: 'rgba(255, 159, 64, 0.1)',
                        tension: 0.3,
                        fill: true
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'top'
                    }},
                    tooltip: {{
                        mode: 'index',
                        intersect: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1.0,
                        ticks: {{
                            callback: function(value) {{
                                return value.toFixed(2);
                            }}
                        }}
                    }}
                }}
            }}
        }});
        
        // Rates Chart
        const ratesCtx = document.getElementById('ratesChart').getContext('2d');
        const ratesChart = new Chart(ratesCtx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(metrics['dates'])},
                datasets: [
                    {{
                        label: 'Hit Rate',
                        data: {json.dumps(metrics['hit_rates'])},
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        tension: 0.3,
                        fill: true
                    }},
                    {{
                        label: 'Rejection Rate',
                        data: {json.dumps(metrics['rejection_rates'])},
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        tension: 0.3,
                        fill: true
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'top'
                    }},
                    tooltip: {{
                        mode: 'index',
                        intersect: false,
                        callbacks: {{
                            label: function(context) {{
                                return context.dataset.label + ': ' + (context.parsed.y * 100).toFixed(1) + '%';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1.0,
                        ticks: {{
                            callback: function(value) {{
                                return (value * 100).toFixed(0) + '%';
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    # 写入文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML report generated: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Generate HTML evaluation report with self-healing review UI")
    parser.add_argument("--data-dir", default="../data", help="Directory containing result JSON files")
    parser.add_argument("--output-dir", default="../reports", help="Output directory for HTML report")
    parser.add_argument("--review-dir", default="review", help="Directory containing review JSON files")
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print("RAG System Report Generator - Iteration 8 (Self-Healing)")
    print(f"{'='*60}")
    
    # 加载所有结果
    results = load_all_results(args.data_dir)
    
    if not results:
        print("❌ No results found. Please run evaluations first.")
        return
    
    # 加载待审核文件
    print(f"\nLoading pending reviews from {args.review_dir}...")
    pending_reviews = load_pending_reviews(args.review_dir)
    print(f"✅ Loaded {len(pending_reviews)} pending reviews")
    
    # 生成报告
    output_path = os.path.join(args.output_dir, "index.html")
    generate_html_report(results, pending_reviews, output_path)
    
    print(f"\n{'='*60}")
    print(f"✅ Report generation complete!")
    print(f"📊 Output: {output_path}")
    if pending_reviews:
        print(f"🔍 Pending reviews: {len(pending_reviews)}")
    print(f"🌐 View in browser or deploy to GitHub Pages")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
