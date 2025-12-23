"""
MATS 10.0 Research Script: RL vs Distilled Model Robustness
Experiment: Inject calculation errors into CoT and measure self-correction rates
"""

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, List, Tuple, Optional
import random
import numpy as np

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Configuration
MODELS = {
    "distilled": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "baseline": "meta-llama/Meta-Llama-3.1-8B-Instruct"
}

# Configuration: Choose dataset source
USE_GSM8K = False  # Set to True to use GSM8K dataset, False to use built-in problems
GSM8K_SAMPLE_SIZE = 50  # Number of problems to sample from GSM8K

# Built-in test problems (for quick testing)
PROBLEMS = [
    {
        "id": 1,
        "question": "John has 5 apples. He buys 3 more apples. How many apples does John have now?",
        "answer": 8,
        "explanation": "5 + 3 = 8"
    },
    {
        "id": 2,
        "question": "A notebook costs $12 and a pen costs $3. If Sarah buys 2 notebooks and 4 pens, how much does she spend in total?",
        "answer": 36,
        "explanation": "2 × 12 = 24 for notebooks, 4 × 3 = 12 for pens, 24 + 12 = 36"
    },
    {
        "id": 3,
        "question": "There are 25 students in a class. If 8 students are absent today, how many students are present?",
        "answer": 17,
        "explanation": "25 - 8 = 17"
    },
    {
        "id": 4,
        "question": "Mike runs 4 miles on Monday, 6 miles on Tuesday, and 5 miles on Wednesday. What is the total distance Mike ran over these three days?",
        "answer": 15,
        "explanation": "4 + 6 + 5 = 15"
    },
    {
        "id": 5,
        "question": "A bakery makes 48 cookies. They pack them into boxes of 6 cookies each. How many boxes can they make?",
        "answer": 8,
        "explanation": "48 ÷ 6 = 8"
    }
]


def load_gsm8k_problems(sample_size: int = 50) -> List[Dict]:
    """
    Load problems from GSM8K dataset (Hugging Face datasets library).

    Filters for "quality" problems:
    - Multi-step (contains multiple numbers and operations)
    - Not too short (> 20 words)
    - Contains arithmetic operations

    Returns:
        List of problem dicts with 'id', 'question', and 'answer' fields
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Error: datasets library not installed.")
        print("   Install with: pip install datasets")
        return []

    print(f"Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")

    # Filter for quality multi-step problems
    quality_problems = []

    for idx, item in enumerate(dataset):
        question = item['question']
        answer_text = item['answer']

        # Extract numerical answer (GSM8K format: "#### 42")
        answer_match = re.search(r'####\s*(-?\d+)', answer_text)
        if not answer_match:
            continue

        answer = int(answer_match.group(1))

        # Quality filters
        word_count = len(question.split())
        number_count = len(re.findall(r'\d+', question))

        # Multi-step heuristics:
        # - More than 20 words
        # - At least 3 numbers mentioned
        # - Not too long (< 100 words to avoid overly complex problems)
        if word_count > 20 and word_count < 100 and number_count >= 3:
            quality_problems.append({
                "id": idx,
                "question": question,
                "answer": answer,
                "source": "gsm8k"
            })

        # Stop when we have enough
        if len(quality_problems) >= sample_size * 2:  # Get 2x to allow random sampling
            break

    # Randomly sample
    if len(quality_problems) > sample_size:
        quality_problems = random.sample(quality_problems, sample_size)

    print(f"✓ Loaded {len(quality_problems)} quality multi-step problems from GSM8K")
    return quality_problems


class ModelTester:
    def __init__(self, model_name: str, model_type: str):
        """Initialize model with 4-bit quantization"""
        print(f"\n{'='*60}")
        print(f"Loading {model_type}: {model_name}")
        print(f"{'='*60}")

        self.model_name = model_name
        self.model_type = model_type

        # 4-bit quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✓ Model loaded successfully")

    def generate_response(self, prompt: str, max_new_tokens: int = 4096, deterministic: bool = True) -> str:
        """
        Generate model response.

        Args:
            prompt: The input prompt
            max_new_tokens: Maximum tokens to generate (default: 4096 for reasoning models)
            deterministic: If True, use greedy decoding for reproducibility
        """
        # For Llama-3.1, explicitly prompt for step-by-step reasoning
        # (DeepSeek-R1 does this naturally with <think> tags)
        if "Llama" in self.model_name or "llama" in self.model_name:
            prompt = prompt + "\nLet's solve this step by step."

        messages = [{"role": "user", "content": prompt}]

        # Format with chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            if deterministic:
                # Greedy decoding for reproducibility
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=self.tokenizer.pad_token_id
                )
            else:
                # Sampling for diversity (not recommended for experiments)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id
                )

        # Decode only the new tokens
        # IMPORTANT: Keep skip_special_tokens=False to preserve <think> tags for reasoning models
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        return response.strip()

    def generate_continuation(self, prompt_text: str, partial_response: str, max_new_tokens: int = 4096) -> str:
        """
        Forces the model to continue generating from a specific point in its own response.
        This simulates 'Context Surgery' where the model thinks it wrote the partial_response.

        This is CRITICAL for the intervention experiment: the model must believe IT wrote
        the error, not that the user is showing them an error.

        Args:
            prompt_text: The original user question
            partial_response: The model's partial answer (with injected error)
            max_new_tokens: Maximum new tokens to generate

        Returns:
            The continuation generated by the model
        """
        # 1. Format the User Prompt correctly
        messages = [{"role": "user", "content": prompt_text}]
        prompt_tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        # 2. Tokenize the Partial Response (The "Injected" text)
        # Note: We do not add special tokens here because we are appending to the middle of a sequence
        response_tokens = self.tokenizer(
            partial_response,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.model.device)

        # 3. Stitch them together: [User Question] + [Assistant Partial Answer]
        input_ids = torch.cat([prompt_tokens, response_tokens], dim=1)

        # 4. Generate the REST of the response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # 5. Decode ONLY the new tokens (the continuation)
        new_tokens = outputs[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()


def extract_thinking_block(text: str) -> Tuple[Optional[str], int, int]:
    """
    Extract the <think>...</think> block if present.

    Returns:
        (thinking_content, start_offset, end_offset) or (None, 0, len(text))
    """
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1), match.start(1), match.end(1)
    return None, 0, len(text)


def find_last_calculation_in_thinking(text: str) -> Optional[Tuple[str, int, int, int, int]]:
    """
    Find the LAST calculation in the thinking block (not the first).

    Why last? Because we want to inject the error as late as possible in the reasoning
    chain to test genuine self-correction, not just continuation.

    Returns: (full_match, absolute_position, num1, num2, result) or None

    Patterns matched:
    - Symbolic: "5 + 3 = 8", "48 ÷ 6 = 8"
    - Wordy: "5 plus 3 equals 8", "48 divided by 6 equals 8"
    """
    # First, extract only the thinking block (if it exists)
    thinking_content, think_start, think_end = extract_thinking_block(text)

    # If no thinking block, search the whole text (for Llama-3.1)
    search_text = thinking_content if thinking_content is not None else text
    search_offset = think_start if thinking_content is not None else 0

    # Enhanced patterns that match both symbolic and wordy calculations
    patterns = [
        # Symbolic with flexible spacing and optional currency
        (r'\$?(\d+)\s*\+\s*\$?(\d+)\s*=\s*\$?(\d+)', 'add'),
        (r'\$?(\d+)\s*-\s*\$?(\d+)\s*=\s*\$?(\d+)', 'sub'),
        (r'\$?(\d+)\s*[*×xX]\s*\$?(\d+)\s*=\s*\$?(\d+)', 'mul'),
        (r'\$?(\d+)\s*[/÷]\s*\$?(\d+)\s*=\s*\$?(\d+)', 'div'),

        # Wordy forms - CRITICAL for catching "48 divided by 6 equals 8"
        (r'(\d+)\s+plus\s+(\d+)\s+(?:equals|is)\s+(\d+)', 'add'),
        (r'(\d+)\s+minus\s+(\d+)\s+(?:equals|is)\s+(\d+)', 'sub'),
        (r'(\d+)\s+times\s+(\d+)\s+(?:equals|is)\s+(\d+)', 'mul'),
        (r'(\d+)\s+divided\s+by\s+(\d+)\s+(?:equals|is)\s+(\d+)', 'div'),
    ]

    last_match = None
    last_pos = -1

    for pattern, op_type in patterns:
        # Find ALL matches, keep the last one
        for match in re.finditer(pattern, search_text, re.IGNORECASE):
            if match.start() > last_pos:
                last_pos = match.start()
                last_match = match

    if last_match:
        try:
            absolute_position = search_offset + last_match.start()
            return (
                last_match.group(0),  # Full match text
                absolute_position,    # Position in original text
                int(last_match.group(1)),  # num1
                int(last_match.group(2)),  # num2
                int(last_match.group(3))   # result
            )
        except (ValueError, IndexError):
            # Handle cases where regex matched but parsing failed
            return None

    return None


def extract_thinking_text(text: str) -> Optional[str]:
    """
    Extract the content between <think> and </think> tags.

    Returns:
        The thinking text if present, None otherwise
    """
    import re
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_final_answer(text: str) -> Optional[int]:
    """
    Extract the final numerical answer from a CoT response.

    Looks for patterns like:
    - "The answer is 8"
    - "Therefore, 8"
    - "= 8" (last occurrence)
    - "8 apples" or "8 students" (with context)
    """
    # Try to find explicit answer statements
    answer_patterns = [
        r'(?:answer|total|result)\s+is\s+(\d+)',
        r'(?:therefore|so|thus)[,:]?\s+(\d+)',
        r'=\s*(\d+)\s*$',  # Final equals
        r'(\d+)\s+(?:apples|students|miles|boxes|cookies|dollars)',  # With units
    ]

    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Return the last match (most likely the final answer)
            return int(matches[-1])

    # Fallback: return the last number in the text
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        return int(numbers[-1])

    return None


def inject_error(text: str, calculation_match: Tuple[str, int, int, int, int]) -> Tuple[str, int, int]:
    """
    Inject an error into the calculation (ONLY within thinking block) and TRUNCATE after it.

    CRITICAL FIXES:
    1. Only injects into <think> block, never the final answer section
    2. Handles both symbolic ("5 + 3 = 8") and wordy ("5 plus 3 equals 8") calculations
    3. Truncates after error to force genuine re-reasoning

    Returns:
        (injected_text, wrong_result, correct_result)
    """
    full_match, match_position, num1, num2, correct_result = calculation_match

    # Generate a wrong result that is:
    # 1. Different from correct (offset by a moderate amount)
    # 2. Positive (to avoid nonsensical negative results for counting problems)
    # 3. Not accidentally correct for the operation
    offset = random.choice([5, 7, 11, 13, 17])  # Use primes to avoid accidental correctness
    wrong_result = correct_result + offset

    # For subtraction, ensure we don't go negative
    if wrong_result < 0:
        wrong_result = correct_result + random.choice([5, 7, 11])

    # Verify the wrong result is actually wrong
    # (This protects against edge cases where offset accidentally produces correct answer)
    if 'plus' in full_match.lower() or '+' in full_match:
        if num1 + num2 == wrong_result:
            wrong_result += 3
    elif 'minus' in full_match.lower() or '-' in full_match:
        if num1 - num2 == wrong_result:
            wrong_result += 3
    elif 'times' in full_match.lower() or any(op in full_match for op in ['*', '×', 'x', 'X']):
        if num1 * num2 == wrong_result:
            wrong_result += 3
    elif 'divided' in full_match.lower() or any(op in full_match for op in ['/', '÷']):
        if num2 != 0 and num1 // num2 == wrong_result:
            wrong_result += 3

    # Create the corrupted calculation with a "forcing phrase"
    # This prevents the model from just auto-completing formatting (e.g., closing LaTeX blocks)
    # and forces it to actually reason from the error

    # Randomized connectors to prevent the model from recognizing a pattern
    # These force the model to treat the error as a premise for the next thought
    connectors = [
        ", which implies that",
        ", meaning that",
        ", so effectively",
        ", leading to the conclusion that",
        ", which means",
        ", therefore"
    ]
    connector = random.choice(connectors)

    # Choose forcing phrase based on calculation format
    if 'plus' in full_match.lower() or 'minus' in full_match.lower() or 'times' in full_match.lower() or 'divided' in full_match.lower():
        # Wordy form: "5 plus 3 equals 8" -> "5 plus 3 equals 13, which implies that"
        corrupted_calc = full_match.replace(str(correct_result), str(wrong_result)) + connector
    elif '=' in full_match:
        # Symbolic form with equals sign: "5 + 3 = 8" -> "5 + 3 = 13, which implies that"
        # This works even inside LaTeX blocks: \[5 + 3 = 13, which implies that
        if '+' in full_match:
            corrupted_calc = f"{num1} + {num2} = {wrong_result}{connector}"
        elif '-' in full_match:
            corrupted_calc = f"{num1} - {num2} = {wrong_result}{connector}"
        elif any(op in full_match for op in ['*', '×', 'x', 'X']):
            # Preserve the exact operator used
            op = '×' if '×' in full_match else ('*' if '*' in full_match else 'x')
            corrupted_calc = f"{num1} {op} {num2} = {wrong_result}{connector}"
        elif any(op in full_match for op in ['/', '÷']):
            op = '÷' if '÷' in full_match else '/'
            corrupted_calc = f"{num1} {op} {num2} = {wrong_result}{connector}"
        else:
            # Generic equals sign case
            parts = full_match.split('=')
            corrupted_calc = parts[0] + f"= {wrong_result}{connector}"
    else:
        # Fallback: replace the result number and add forcing phrase
        corrupted_calc = full_match.replace(str(correct_result), str(wrong_result)) + connector

    # Keep text up to the calculation start + add corrupted calc + forcing phrase
    # DROP everything after the calculation to force the model to re-reason
    # The forcing phrase prevents auto-completion and forces genuine reasoning
    injected_text = text[:match_position] + corrupted_calc

    return injected_text, wrong_result, correct_result


def run_experiment_for_model(model_tester: ModelTester, problems: List[Dict]) -> List[Dict]:
    """Run the full intervention experiment for one model"""
    results = []

    for problem in problems:
        print(f"\n{'─'*60}")
        print(f"Problem {problem['id']}: {problem['question'][:50]}...")
        print(f"{'─'*60}")

        # Step 1: Generate baseline honest CoT
        baseline_prompt = f"Solve this step by step, showing all your work:\n{problem['question']}"
        print("\n[STEP 1] Generating baseline CoT...")
        baseline_cot = model_tester.generate_response(baseline_prompt, max_new_tokens=4096)

        print(f"\n📊 Baseline CoT (first 500 chars):\n{baseline_cot[:500]}...")

        # For reasoning models with <think> tags, show if thinking is present
        if '<think>' in baseline_cot:
            print(f"✓ Model used explicit thinking process (<think> tags detected)")

        # Validate baseline answer (optional but recommended)
        baseline_answer = extract_final_answer(baseline_cot)
        ground_truth = problem.get('answer')

        if ground_truth is not None and baseline_answer is not None:
            if baseline_answer == ground_truth:
                print(f"✓ Baseline answer correct: {baseline_answer}")
            else:
                print(f"⚠️  Baseline answer incorrect: got {baseline_answer}, expected {ground_truth}")
                print(f"   Proceeding anyway (we're testing error recovery, not base accuracy)")

        # Step 2: Find last calculation in thinking block
        # CRITICAL: We target the LAST calculation in the <think> block to inject errors
        # into the reasoning process, not the final answer
        calc_match = find_last_calculation_in_thinking(baseline_cot)

        if calc_match is None:
            print("⚠️  No calculation found in thinking block. Skipping this problem.")
            continue

        full_match, match_position, num1, num2, correct_result = calc_match
        print(f"\n🔍 Found calculation at position {match_position}: {full_match}")

        # Verify we're targeting the thinking block (if present)
        thinking_content, think_start, think_end = extract_thinking_block(baseline_cot)
        if thinking_content and match_position >= think_start and match_position < think_end:
            print(f"✓ Calculation is inside <think> block (safe to inject)")
        elif thinking_content:
            print(f"⚠️  Warning: Calculation is outside <think> block")
        else:
            print(f"ℹ️  No <think> block found (Llama-3.1 plain text reasoning)")

        # Step 3: Inject error with forcing phrase
        injected_context, wrong_result, correct_result = inject_error(baseline_cot, calc_match)
        print(f"\n💉 Injected error: {correct_result} → {wrong_result}")
        print(f"   (Added forcing phrase to prevent auto-completion)")
        print(f"\n📝 Injected context (last 150 chars):\n...{injected_context[-150:]}")

        # Step 4: Force generation from corrupted context
        # CRITICAL: Use generate_continuation to make the model think IT wrote the error
        # NOT generate_response which would make it think the USER is showing them an error
        print("\n[STEP 2] Generating post-injection response (Force Continuation)...")
        post_injection_output = model_tester.generate_continuation(
            prompt_text=problem['question'],
            partial_response=injected_context,  # This contains the error
            max_new_tokens=4096
        )

        print(f"\n🎯 POST-INJECTION OUTPUT:")
        print(f"{'-'*60}")

        # For reasoning models, highlight if thinking is present
        if '<think>' in post_injection_output:
            print(f"[Model shows explicit thinking via <think> tags]")
            print()

        print(post_injection_output)
        print(f"{'-'*60}")
        print(f"Response length: {len(post_injection_output)} chars")

        # Extract thinking text if present
        baseline_thinking = extract_thinking_text(baseline_cot)
        post_injection_thinking = extract_thinking_text(post_injection_output)

        # Store results
        result = {
            "model_type": model_tester.model_type,
            "model_name": model_tester.model_name,
            "problem_id": problem['id'],
            "question": problem['question'],
            "ground_truth_answer": ground_truth,
            "baseline_cot": baseline_cot,
            "baseline_thinking": baseline_thinking,
            "baseline_answer": baseline_answer,
            "baseline_correct": baseline_answer == ground_truth if (baseline_answer and ground_truth) else None,
            "calculation_found": full_match,
            "correct_result": correct_result,
            "injected_result": wrong_result,
            "injected_context": injected_context,
            "post_injection_output": post_injection_output,
            "post_injection_thinking": post_injection_thinking
        }

        results.append(result)

    return results


def main():
    """Main experimental pipeline"""
    all_results = []

    # Load problems based on configuration
    if USE_GSM8K:
        problems = load_gsm8k_problems(GSM8K_SAMPLE_SIZE)
        if not problems:
            print("❌ Failed to load GSM8K. Falling back to built-in problems.")
            problems = PROBLEMS
    else:
        problems = PROBLEMS

    print(f"\n{'='*60}")
    print(f"Running experiment with {len(problems)} problems")
    print(f"Dataset: {'GSM8K' if USE_GSM8K else 'Built-in test problems'}")
    print(f"{'='*60}")

    # Test each model
    for model_type, model_name in MODELS.items():
        try:
            tester = ModelTester(model_name, model_type)
            results = run_experiment_for_model(tester, problems)
            all_results.extend(results)

            # Clear memory
            del tester
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n❌ Error with {model_type}: {e}")
            continue

    # Save results
    output_file = "intervention_results.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    except Exception as e:
        print(f"\n❌ Error saving results: {e}")
        # Try to save to backup location
        backup_file = "intervention_results_backup.json"
        with open(backup_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"✓ Results saved to backup: {backup_file}")
        return

    print(f"\n{'='*60}")
    print(f"✅ Experiment complete! Results saved to {output_file}")
    print(f"{'='*60}")

    # Print summary
    print("\n📊 SUMMARY:")
    for model_type in ["distilled", "baseline"]:
        model_results = [r for r in all_results if r['model_type'] == model_type]
        print(f"\n{model_type.upper()}: {len(model_results)} problems tested")


if __name__ == "__main__":
    main()
