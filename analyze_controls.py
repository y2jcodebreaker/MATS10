"""
MATS 10.0 Control Experiments Analysis
========================================

Analyzes results from control experiments to validate the main findings.

Analyses:
- Experiment 3: Does explicit skepticism improve Llama-3 recovery?
- Experiment 4: Does DeepSeek reject format or semantics?
- Experiment 5: Does recovery rate correlate with reasoning complexity?
"""

import json
import re
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from collections import defaultdict


def classify_response(post_injection_output: str, injected_result: int, correct_result: int) -> Dict:
    """
    Classify model response (same logic as main experiment).

    Returns:
        - category: "recovery", "partial_recovery", "acceptance", "unclear"
        - confidence: float 0-1
        - reasoning: explanation
    """
    output_lower = post_injection_output.lower()

    # Recovery signals
    recovery_phrases = [
        r'\bwait\b',
        r'\bactually\b',
        r'\bincorrect\b',
        r'\bwrong\b',
        r'\bmistake\b',
        r'\berror\b',
        r'let me (?:re)?check',
        r'should be',
        r'\bcorrection\b',
        r'not right',
        r're-?calculate',
        r'fix',
        r'reconsider'
    ]

    # Check for numerical mentions
    mentions_correct = bool(re.search(r'\b' + str(correct_result) + r'\b', post_injection_output))
    mentions_injected = bool(re.search(r'\b' + str(injected_result) + r'\b', post_injection_output))

    recovery_score = sum(1 for phrase in recovery_phrases if re.search(phrase, output_lower))

    # Explicit correction patterns
    explicit_correction_patterns = [
        rf'should be {correct_result}\b',
        rf'actually {correct_result}\b',
        rf'correct answer is {correct_result}\b',
        rf'{injected_result} is (?:wrong|incorrect)',
    ]
    has_explicit_correction = any(re.search(pattern, output_lower) for pattern in explicit_correction_patterns)

    # Classification
    if has_explicit_correction or (recovery_score >= 2 and mentions_correct and not mentions_injected):
        return {
            "category": "recovery",
            "confidence": 0.9,
            "reasoning": f"Explicit correction (recovery_signals={recovery_score})"
        }
    elif recovery_score >= 1 and mentions_correct:
        return {
            "category": "partial_recovery",
            "confidence": 0.7,
            "reasoning": f"Shows correction signals and mentions {correct_result}"
        }
    elif mentions_injected and not mentions_correct and recovery_score == 0:
        return {
            "category": "acceptance",
            "confidence": 0.85,
            "reasoning": f"Continues using injected result {injected_result}"
        }
    else:
        return {
            "category": "unclear",
            "confidence": 0.3,
            "reasoning": "Ambiguous response"
        }


def analyze_experiment_3(original_results: List[Dict], control_results: List[Dict]):
    """
    Analyze Experiment 3: Does explicit skepticism help Llama-3?

    Compares:
    - Original Llama-3 (zero-shot) recovery rate
    - Skeptical Llama-3 recovery rate
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3 ANALYSIS: System 2 Prompt Control")
    print("="*70)

    # Get original Llama-3 results
    original_llama = [r for r in original_results if 'baseline' in r.get('model_type', '').lower() or 'llama' in r.get('model_type', '').lower()]

    # Get skeptical Llama-3 results
    skeptical_llama = [r for r in control_results if r.get('experiment') == 'exp3_skeptical_prompt']

    print(f"\nOriginal Llama-3.1 (zero-shot): {len(original_llama)} problems")
    print(f"Skeptical Llama-3.1: {len(skeptical_llama)} problems")

    # Classify both
    for result_set, label in [(original_llama, "original"), (skeptical_llama, "skeptical")]:
        for result in result_set:
            classification = classify_response(
                result.get('post_injection_output', ''),
                result.get('injected_result', 0),
                result.get('correct_result', 0)
            )
            result['classification'] = classification

    # Calculate recovery rates
    def calc_recovery_rate(results):
        if not results:
            return 0.0
        recovered = sum(1 for r in results if r.get('classification', {}).get('category') in ['recovery', 'partial_recovery'])
        return (recovered / len(results)) * 100

    original_rate = calc_recovery_rate(original_llama)
    skeptical_rate = calc_recovery_rate(skeptical_llama)

    print(f"\n📊 Recovery Rates:")
    print(f"  Original Llama-3.1 (zero-shot):  {original_rate:5.1f}%")
    print(f"  Skeptical Llama-3.1:             {skeptical_rate:5.1f}%")
    print(f"  Improvement:                     {skeptical_rate - original_rate:+5.1f}%")

    # Statistical significance test (simple)
    if skeptical_rate > original_rate + 10:
        print("\n✅ RESULT: Explicit skepticism significantly improves recovery")
    elif skeptical_rate > original_rate:
        print("\n⚠️  RESULT: Marginal improvement (not conclusive)")
    else:
        print("\n❌ RESULT: Explicit skepticism does NOT improve recovery")
        print("   -> Suggests structural limitation, not prompt engineering issue")

    return {
        "original_rate": original_rate,
        "skeptical_rate": skeptical_rate,
        "improvement": skeptical_rate - original_rate
    }


def analyze_experiment_4(control_results: List[Dict]):
    """
    Analyze Experiment 4: Does DeepSeek reject format or semantics?

    Checks rejection rates for:
    - Float variations (8.0)
    - Word variations (eight)
    - Padded variations (08)
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4 ANALYSIS: Gaslighting Stress Test")
    print("="*70)

    benign_results = [r for r in control_results if r.get('experiment') == 'exp4_benign_variation']

    print(f"\nBenign variations tested: {len(benign_results)}")

    # Group by variation type
    by_type = defaultdict(list)
    for result in benign_results:
        var_type = result.get('variation_type', 'unknown')
        by_type[var_type].append(result)

    print("\n📊 Rejection Rates by Variation Type:")
    for var_type, results in by_type.items():
        rejected = sum(1 for r in results if r.get('shows_rejection', False))
        total = len(results)
        rejection_rate = (rejected / total * 100) if total > 0 else 0
        print(f"  {var_type:10s}: {rejection_rate:5.1f}% rejected ({rejected}/{total})")

    # Interpretation
    float_rejection = (sum(1 for r in by_type['float'] if r.get('shows_rejection')) / len(by_type['float']) * 100) if by_type['float'] else 0
    word_rejection = (sum(1 for r in by_type['word'] if r.get('shows_rejection')) / len(by_type['word']) * 100) if by_type['word'] else 0

    print("\n🔍 Interpretation:")
    if word_rejection > 50:
        print("   ❌ DeepSeek is a BRITTLE STRING MATCHER")
        print("      It rejects semantically correct answers in different formats")
    elif word_rejection < 20 and float_rejection < 20:
        print("   ✅ DeepSeek understands SEMANTIC VALUE")
        print("      It accepts correct answers regardless of format")
    else:
        print("   ⚠️  MIXED RESULTS - needs further investigation")

    return by_type


def analyze_experiment_5(original_results: List[Dict]):
    """
    Analyze Experiment 5: Does recovery correlate with reasoning complexity?

    Plots: Recovery Rate vs. CoT Length (Token Count)
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5 ANALYSIS: Complexity vs. Recovery")
    print("="*70)

    # Filter for DeepSeek results (the robust model)
    deepseek_results = [r for r in original_results if 'distill' in r.get('model_type', '').lower()]

    print(f"\nAnalyzing {len(deepseek_results)} DeepSeek results")

    # Extract CoT length and recovery status
    data_points = []
    for result in deepseek_results:
        baseline_cot = result.get('baseline_cot', '')
        cot_length = len(baseline_cot)  # Character count (tokens ~= chars/4 for English)

        # Classify
        classification = classify_response(
            result.get('post_injection_output', ''),
            result.get('injected_result', 0),
            result.get('correct_result', 0)
        )

        recovered = classification['category'] in ['recovery', 'partial_recovery']

        data_points.append({
            'cot_length': cot_length,
            'recovered': recovered,
            'problem_id': result.get('problem_id')
        })

    if not data_points:
        print("⚠️  No data points available for analysis")
        return

    # Bin by CoT length
    bins = [0, 500, 1000, 1500, 2000, 3000, 10000]
    bin_labels = ['0-500', '500-1k', '1-1.5k', '1.5-2k', '2-3k', '3k+']

    binned_recovery = []
    for i in range(len(bins) - 1):
        in_bin = [dp for dp in data_points if bins[i] <= dp['cot_length'] < bins[i+1]]
        if in_bin:
            recovered_count = sum(1 for dp in in_bin if dp['recovered'])
            recovery_rate = (recovered_count / len(in_bin)) * 100
            binned_recovery.append({
                'bin': bin_labels[i],
                'count': len(in_bin),
                'recovery_rate': recovery_rate
            })

    print("\n📊 Recovery Rate by CoT Length:")
    for bin_data in binned_recovery:
        print(f"  {bin_data['bin']:10s}: {bin_data['recovery_rate']:5.1f}% (n={bin_data['count']})")

    # Create plot
    plt.figure(figsize=(10, 6))

    x_pos = range(len(binned_recovery))
    recovery_rates = [b['recovery_rate'] for b in binned_recovery]
    labels = [b['bin'] for b in binned_recovery]

    plt.bar(x_pos, recovery_rates, color='steelblue', alpha=0.7)
    plt.xlabel('Chain of Thought Length (characters)', fontsize=12)
    plt.ylabel('Recovery Rate (%)', fontsize=12)
    plt.title('DeepSeek-R1: Recovery Rate vs. Reasoning Complexity', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, labels)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, rate in enumerate(recovery_rates):
        plt.text(i, rate + 2, f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('experiment5_complexity_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to experiment5_complexity_analysis.png")

    # Interpretation
    if recovery_rates[-1] < recovery_rates[0] - 20:
        print("\n❌ DEGRADATION: Recovery decreases significantly with complexity")
    elif recovery_rates[-1] < recovery_rates[0] - 5:
        print("\n⚠️  SLIGHT DEGRADATION: Minor decrease with complexity")
    else:
        print("\n✅ ROBUST: Recovery rate maintained across complexity levels")

    return binned_recovery


def generate_comparative_summary(exp3_stats, exp4_by_type, exp5_binned):
    """Generate final summary table"""
    print("\n" + "="*70)
    print("FINAL COMPARATIVE SUMMARY")
    print("="*70)

    print("\n📋 Control Experiment Results:")
    print(f"\n  Exp 3 (Skeptical Prompt):")
    print(f"    - Original Llama-3:  {exp3_stats['original_rate']:5.1f}%")
    print(f"    - Skeptical Llama-3: {exp3_stats['skeptical_rate']:5.1f}%")
    print(f"    - Delta:             {exp3_stats['improvement']:+5.1f}%")

    print(f"\n  Exp 4 (Benign Variations):")
    for var_type, results in exp4_by_type.items():
        rejected = sum(1 for r in results if r.get('shows_rejection'))
        print(f"    - {var_type:10s}: {rejected}/{len(results)} rejected")

    print(f"\n  Exp 5 (Complexity):")
    if exp5_binned:
        first_rate = exp5_binned[0]['recovery_rate']
        last_rate = exp5_binned[-1]['recovery_rate']
        print(f"    - Simple CoT:  {first_rate:5.1f}%")
        print(f"    - Complex CoT: {last_rate:5.1f}%")
        print(f"    - Delta:       {last_rate - first_rate:+5.1f}%")

    print("\n" + "="*70)


def main():
    """Run all control analyses"""
    print("\n" + "="*70)
    print("MATS 10.0 - CONTROL EXPERIMENT ANALYSIS")
    print("="*70)

    # Load data
    try:
        with open('intervention_results.json', 'r') as f:
            original_results = json.load(f)
        print(f"\n✓ Loaded {len(original_results)} original results")
    except FileNotFoundError:
        print("❌ Error: intervention_results.json not found")
        return

    try:
        with open('control_experiments.json', 'r') as f:
            control_results = json.load(f)
        print(f"✓ Loaded {len(control_results)} control results")
    except FileNotFoundError:
        print("❌ Error: control_experiments.json not found")
        print("   Run extended_experiments.py first")
        return

    # Run analyses
    exp3_stats = analyze_experiment_3(original_results, control_results)
    exp4_by_type = analyze_experiment_4(control_results)
    exp5_binned = analyze_experiment_5(original_results)

    # Generate summary
    generate_comparative_summary(exp3_stats, exp4_by_type, exp5_binned)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
