"""
完整评估脚本：扫描 /evaluation 下所有 sample_* 目录，计算所有 6 个指标，输出 JSON。
使用: python evaluate_all.py [--output-dir results]
"""

import os
import json
import glob
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

# 替换原有 clap 导入，改用新的 CLAPEvaluator
from metrics.clap import CLAPEvaluator  # 新的CLAP类
from metrics.acoustic_similarity import compute_pairwise_cosine as compute_acoustic_similarity # 导入物理相似度

# 导入其他指标模块（不变）
from metrics.fad import compute_fad
from metrics.js_kl import compute_js_kl
from metrics.ndb import compute_ndb
from metrics.beat import compute_beat_metrics


def scan_evaluation_dir(eval_root):
    """扫描 evaluation 目录，返回 (样本号, gt_path, gen_path) 列表。"""
    samples = []
    sample_dirs = sorted(glob.glob(os.path.join(eval_root, "sample_*")))
    for d in sample_dirs:
        gt_path = os.path.join(d, "gt.wav")
        gen_path = os.path.join(d, "gen.wav")
        if os.path.exists(gt_path) and os.path.exists(gen_path):
            sample_id = os.path.basename(d)
            samples.append((sample_id, gt_path, gen_path))
    return samples


def evaluate_single(gt_path, gen_path, clap_evaluator):
    """对单个样本计算所有指标（新增 clap_evaluator 参数）。"""
    result = {"gt": gt_path, "gen": gen_path}
    
    # 注意：FAD / JS-KL / NDB 是全量集合级别指标，不适合单样本计算。
    # 在 per-sample 结果中保留占位（由 batch 计算填充），避免产生误导性的恒定值。
    result["fad"] = None
    result["fad_note"] = "batch-only"
    result["js_mean"] = None
    result["kl_mean"] = None
    result["jskl_note"] = "batch-only"
    result["ndb"] = None
    result["ndb_note"] = "batch-only"
    
    # 物理音色相似度 (Acoustic Similarity - 原MFCC逻辑)
    try:
        # 因为你的脚本里默认就是MFCC，直接传入路径即可
        acoustic_result = compute_acoustic_similarity([gt_path], [gen_path])
        result["acoustic_similarity"] = float(acoustic_result["per_sample"][0])
    except Exception as e:
        result["acoustic_similarity"] = None


    # 真实CLAP语义相似度（替换原MFCC余弦相似度）
    try:
        # 单样本调用CLAP评估器
        clap_result = clap_evaluator.compute_metrics([gt_path], [gen_path])
        result["cosine_similarity"] = float(clap_result["per_sample"][0])
        result["clap_type"] = "LAION-CLAP (semantic embedding)"  # 标注CLAP类型，避免误解
    except Exception as e:
        result["cosine_similarity"] = None
        result["clap_error"] = str(e)
    
    # Beat(节拍匹配)
    try:
        beat_result = compute_beat_metrics([gt_path], [gen_path])
        result["beat_f1"] = float(beat_result["per_sample_f1"][0])
        result["beat_precision"] = float(beat_result["per_sample_precision"][0])
        result["beat_recall"] = float(beat_result["per_sample_recall"][0])
        result["beat_error"] = float(beat_result["per_sample_err"][0])
    except Exception as e:
        result["beat_f1"] = None
        result["beat_precision"] = None
        result["beat_recall"] = None
        result["beat_error"] = None
        result["beat_error_msg"] = str(e)
    
    # VA (需要用户提供真实 VA 标签，这里跳过)
    result["va_distance"] = None
    result["va_cosine"] = None
    result["va_status"] = "需要用户提供 VA 标签"
    
    return result


def evaluate_batch(gt_list, gen_list, clap_evaluator):
    """对全量样本批量计算指标（新增 clap_evaluator 参数，可选批量CLAP评估）。"""
    results = {}
    
    # FAD (整体)
    try:
        fad_val, _ = compute_fad(gt_list, gen_list)
        results["fad_overall"] = float(fad_val)
    except Exception as e:
        results["fad_overall"] = None
        results["fad_overall_error"] = str(e)
    
    # NDB (整体)
    try:
        ndb_result = compute_ndb(gt_list, gen_list, K=50)
        results["ndb_overall"] = int(ndb_result["ndb"])
        results["ndb_K"] = 50
    except Exception as e:
        results["ndb_overall"] = None
        results["ndb_overall_error"] = str(e)
    
    # JS/KL (整体)
    try:
        js_kl = compute_js_kl(gt_list, gen_list)
        results["js_kl_overall"] = {
            "js_mean": float(js_kl["js_mean"]),
            "kl_mean": float(js_kl["kl_mean"])
        }
    except Exception as e:
        results["js_kl_overall"] = None
        results["js_kl_overall_error"] = str(e)
    
    # 可选：批量计算CLAP（提升效率，单样本已算过，这里可省略）
    # try:
    #     clap_batch_result = clap_evaluator.compute_metrics(gt_list, gen_list)
    #     results["clap_mean_batch"] = float(clap_batch_result["mean"])
    #     results["clap_std_batch"] = float(clap_batch_result["std"])
    # except Exception as e:
    #     results["clap_mean_batch"] = None
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="/lm2d/evaluation", help="评估目录路径")
    parser.add_argument("--output-dir", default="/lm2d/results", help="输出结果目录")
    args = parser.parse_args()
    
    eval_root = args.eval_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 扫描样本
    print(f"扫描 {eval_root} 目录...")
    samples = scan_evaluation_dir(eval_root)
    print(f"找到 {len(samples)} 个样本")
    
    if not samples:
        print("错误: 未找到任何样本")
        return
    
    print("\n初始化CLAP模型...")
    try:
        clap_evaluator = CLAPEvaluator()  # 自动选择cuda/cpu
        print("CLAP模型初始化成功！")
    except Exception as e:
        print(f"CLAP模型初始化失败: {str(e)}")
        print("将退出评估（CLAP是核心指标）")
        return
    
    # 逐样本评估
    print("\n逐样本评估...")
    sample_results = {}
    gt_list = []
    gen_list = []
    
    for sample_id, gt_path, gen_path in tqdm(samples, desc="样本评估进度"):
        # 传入CLAP评估器
        sample_results[sample_id] = evaluate_single(gt_path, gen_path, clap_evaluator)
        gt_list.append(gt_path)
        gen_list.append(gen_path)
    
    # 全量评估（传入CLAP评估器，可选批量CLAP）
    print("\n全量指标评估...")
    batch_results = evaluate_batch(gt_list, gen_list, clap_evaluator)
    
    # 聚合每样本指标的平均值

    # 聚合平均值
    acoustic_vals = [r["acoustic_similarity"] for r in sample_results.values() if r["acoustic_similarity"] is not None]
    acoustic_mean = float(np.mean(acoustic_vals)) if acoustic_vals else None

    #clap_vals = [r["clap_cosine_similarity"] for r in sample_results.values() if r["clap_cosine_similarity"] is not None]
    #clap_mean = float(np.mean(clap_vals)) if clap_vals else None

    cosine_vals = [r["cosine_similarity"] for r in sample_results.values() if r["cosine_similarity"] is not None]
    cosine_mean = float(np.mean(cosine_vals)) if cosine_vals else None

    beat_f1_vals = [r["beat_f1"] for r in sample_results.values() if r["beat_f1"] is not None]
    beat_f1_mean = float(np.mean(beat_f1_vals)) if beat_f1_vals else None

    beat_precision_vals = [r["beat_precision"] for r in sample_results.values() if r["beat_precision"] is not None]
    beat_precision_mean = float(np.mean(beat_precision_vals)) if beat_precision_vals else None

    beat_recall_vals = [r["beat_recall"] for r in sample_results.values() if r["beat_recall"] is not None]
    beat_recall_mean = float(np.mean(beat_recall_vals)) if beat_recall_vals else None

    beat_error_vals = [r["beat_error"] for r in sample_results.values() if r["beat_error"] is not None]
    beat_error_mean = float(np.mean(beat_error_vals)) if beat_error_vals else None

    # 将关键的 batch 指标放入 metadata（并保留 batch_metrics 字段以向后兼容）
    metadata = {
        "total_samples": len(samples),
        "eval_dir": eval_root,
        "acoustic_similarity_mean": acoustic_mean,
        "beat_precision_mean": beat_precision_mean,
        "beat_recall_mean": beat_recall_mean,
        "beat_error_mean": beat_error_mean
    }
    if batch_results.get("fad_overall") is not None:
        metadata["fad_overall"] = float(batch_results["fad_overall"])
    if batch_results.get("js_kl_overall"):
        metadata["js_kl_overall"] = batch_results["js_kl_overall"]
    if batch_results.get("ndb_overall") is not None:
        metadata["ndb_overall"] = int(batch_results["ndb_overall"])
    if batch_results.get("ndb_K") is not None:
        metadata["ndb_K"] = int(batch_results["ndb_K"])
    if beat_f1_mean is not None:
        metadata["beat_F1"] = float(beat_f1_mean)
    if cosine_mean is not None:
        metadata["clap_mean"] = float(cosine_mean)
        metadata["clap_type"] = "LAION-CLAP (semantic embedding)"  # 标注CLAP类型


    final_result = {
        "metadata": metadata,
        "batch_metrics": batch_results,
        "per_sample_metrics": sample_results
    }
    
    # 保存 JSON
    output_file = os.path.join(output_dir, "evaluation_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 结果已保存到 {output_file}")
    
    # 打印摘要
    print("\n========== 指标摘要 ==========")
    print(f"总样本数: {len(samples)}")
    if batch_results.get("fad_overall"):
        print(f"FAD (整体): {batch_results['fad_overall']:.4f}")
    if batch_results.get("ndb_overall"):
        print(f"NDB (整体): {batch_results['ndb_overall']}")
    if batch_results.get("js_kl_overall"):
        print(f"JS/KL - JS: {batch_results['js_kl_overall']['js_mean']:.4f}, KL: {batch_results['js_kl_overall']['kl_mean']:.4f}")
    if acoustic_mean:
        print(f"Acoustic Similarity (MFCC): {acoustic_mean:.4f}")
    if cosine_mean:
        print(f"Semantic Similarity (CLAP): {cosine_mean:.4f}")
    
    # 聚合每样本指标的平均值
    cosine_vals = [r["cosine_similarity"] for r in sample_results.values() if r["cosine_similarity"] is not None]
    if cosine_vals:
        print(f"CLAP余弦相似度 (平均): {np.mean(cosine_vals):.4f} (LAION-CLAP语义嵌入)")
    
    beat_f1_vals = [r["beat_f1"] for r in sample_results.values() if r["beat_f1"] is not None]
    if beat_f1_vals:
        print(f"Beat F1 (平均): {np.mean(beat_f1_vals):.4f}")


if __name__ == "__main__":
    main()
#python evaluate_all.py --eval-dir evaluation --output-dir results