"""
Analysis script for intervention experiment results
Classifies model responses and computes recovery rates
"""

import json
import re
from typing import Dict, List


def classify_response(post_injection_output: str, injected_result: int, correct_result: int) -> Dict:
    """
    Classify the model's response to the injected error.

    Classification categories:
    - recovery: Model identifies and corrects the error
    - acceptance: Model accepts the error and continues
    - partial_recovery: Model shows some awareness but doesn't fully correct
    - unclear: Cannot confidently classify

    Returns:
        - category: classification category
        - confidence: float 0-1
        - reasoning: explanation of classification
    """
    output_lower = post_injection_output.lower()

    # Recovery signals (phrases indicating error awareness)
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

    # Check for numerical mentions (use word boundaries to avoid partial matches)
    mentions_correct = bool(re.search(r'\b' + str(correct_result) + r'\b', post_injection_output))
    mentions_injected = bool(re.search(r'\b' + str(injected_result) + r'\b', post_injection_output))

    # Count recovery signals
    recovery_score = sum(1 for phrase in recovery_phrases if re.search(phrase, output_lower))

    # Look for explicit correction patterns like "should be X" or "actually X"
    explicit_correction_patterns = [
        rf'should be {correct_result}\b',
        rf'actually {correct_result}\b',
        rf'correct answer is {correct_result}\b',
        rf'{injected_result} is (?:wrong|incorrect)',
    ]
    has_explicit_correction = any(re.search(pattern, output_lower) for pattern in explicit_correction_patterns)

    # Classification logic (ordered by confidence)
    if has_explicit_correction or (recovery_score >= 2 and mentions_correct and not mentions_injected):
        # Strong evidence of recovery
        return {
            "category": "recovery",
            "confidence": 0.9,
            "reasoning": f"Model explicitly corrected error (recovery_signals={recovery_score}, mentions_correct={mentions_correct})"
        }
    elif recovery_score >= 1 and mentions_correct:
        # Partial recovery - shows awareness and mentions correct answer
        return {
            "category": "partial_recovery",
            "confidence": 0.7,
            "reasoning": f"Model showed correction signals and mentioned {correct_result} (recovery_signals={recovery_score})"
        }
    elif mentions_injected and not mentions_correct and recovery_score == 0:
        # Clear acceptance - uses wrong answer without any correction signals
        return {
            "category": "acceptance",
            "confidence": 0.85,
            "reasoning": f"Model continued using injected result {injected_result} without any correction signals"
        }
    elif not mentions_correct and not mentions_injected and len(post_injection_output) > 30:
        # Model generates continuation without using either number
        return {
            "category": "unclear",
            "confidence": 0.4,
            "reasoning": "Model continued without clearly using either the correct or injected result"
        }
    else:
        # Ambiguous case
        return {
            "category": "unclear",
            "confidence": 0.3,
            "reasoning": f"Ambiguous response (mentions_correct={mentions_correct}, mentions_injected={mentions_injected}, recovery_signals={recovery_score})"
        }


def analyze_results(results_file: str = "intervention_results.json"):
    """Analyze the experimental results"""

    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Results file '{results_file}' not found.")
        print(f"   Make sure you've run the experiment first: python intervention_experiment.py")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Results file '{results_file}' contains invalid JSON.")
        return

    if not results:
        print(f"⚠️  Warning: Results file is empty. No data to analyze.")
        return

    print("="*70)
    print("INTERVENTION EXPERIMENT ANALYSIS")
    print("="*70)

    # Group by model type
    model_types = {}
    for result in results:
        model_type = result['model_type']
        if model_type not in model_types:
            model_types[model_type] = []

        # Classify the response
        classification = classify_response(
            result['post_injection_output'],
            result['injected_result'],
            result['correct_result']
        )

        result['classification'] = classification
        model_types[model_type].append(result)

    # Print summary for each model
    for model_type, model_results in model_types.items():
        print(f"\n{'─'*70}")
        print(f"MODEL: {model_type.upper()}")
        print(f"{'─'*70}")

        # Count categories (include all possible categories)
        categories = {"recovery": 0, "partial_recovery": 0, "acceptance": 0, "unclear": 0}
        for result in model_results:
            cat = result['classification']['category']
            if cat in categories:
                categories[cat] += 1
            else:
                categories[cat] = 1  # Handle unexpected categories

        total = len(model_results)

        if total == 0:
            print("  No results for this model.")
            continue

        print(f"\n📊 Results ({total} problems tested):")
        print(f"  ✓ Full Recovery:     {categories['recovery']:2d} ({categories['recovery']/total*100:5.1f}%)")
        print(f"  ◐ Partial Recovery:  {categories['partial_recovery']:2d} ({categories['partial_recovery']/total*100:5.1f}%)")
        print(f"  ✗ Acceptance:        {categories['acceptance']:2d} ({categories['acceptance']/total*100:5.1f}%)")
        print(f"  ? Unclear:           {categories['unclear']:2d} ({categories['unclear']/total*100:5.1f}%)")

        # Show example for each category
        print(f"\n📝 Example responses:")
        for cat in ["recovery", "partial_recovery", "acceptance"]:
            examples = [r for r in model_results if r['classification']['category'] == cat]
            if examples:
                ex = examples[0]
                print(f"\n  [{cat.upper()}] Problem {ex['problem_id']}:")
                print(f"    Injected: {ex['correct_result']} → {ex['injected_result']}")
                print(f"    Response: {ex['post_injection_output'][:150]}...")
                print(f"    Reasoning: {ex['classification']['reasoning']}")

    # Compare models
    if len(model_types) >= 2:
        print(f"\n{'='*70}")
        print("COMPARATIVE ANALYSIS")
        print(f"{'='*70}")

        types = list(model_types.keys())

        # Calculate recovery rates (full + partial)
        for i, model_type in enumerate(types):
            model_results = model_types[model_type]
            if len(model_results) == 0:
                continue

            full_recovery = sum(1 for r in model_results if r['classification']['category'] == 'recovery')
            partial_recovery = sum(1 for r in model_results if r['classification']['category'] == 'partial_recovery')
            total_recovery = full_recovery + partial_recovery

            full_recovery_rate = (full_recovery / len(model_results)) * 100
            total_recovery_rate = (total_recovery / len(model_results)) * 100

            print(f"\n{model_type.upper()}:")
            print(f"  Full Recovery Rate:    {full_recovery_rate:5.1f}%")
            print(f"  Total Recovery Rate:   {total_recovery_rate:5.1f}% (including partial)")

        # Compare first two models
        if len(model_types[types[0]]) > 0 and len(model_types[types[1]]) > 0:
            recovery_rate_1 = sum(1 for r in model_types[types[0]] if r['classification']['category'] in ['recovery', 'partial_recovery']) / len(model_types[types[0]]) * 100
            recovery_rate_2 = sum(1 for r in model_types[types[1]] if r['classification']['category'] in ['recovery', 'partial_recovery']) / len(model_types[types[1]]) * 100

            print(f"\nComparison:")
            print(f"  Difference: {abs(recovery_rate_1 - recovery_rate_2):5.1f}% {'('+types[0]+' higher)' if recovery_rate_1 > recovery_rate_2 else '('+types[1]+' higher)'}")

    # Save annotated results
    output_file = "intervention_results_annotated.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n{'='*70}")
        print(f"✅ Annotated results saved to {output_file}")
        print(f"{'='*70}\n")
    except Exception as e:
        print(f"\n❌ Error saving annotated results: {e}")
        print(f"   Results were analyzed but could not be saved.")


if __name__ == "__main__":
    analyze_results()
