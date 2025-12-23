# The Fragility of Chain-of-Thought: Epistemic Robustness in Distilled Reasoning Models

**MATS 10.0 Application Project (Winter 2024/25)**
*Model Biology & Applied Interpretability Track*

## 🧬 Project Overview
This project investigates a critical question in Model Biology: **Does "Reasoning" training create a genuine internal Truth Anchor, or is it merely a stylistic imitation?**

We benchmark the **"Epistemic Stubbornness"** of Distilled Reasoning Models (DeepSeek-R1-Distill-Llama-8B) against Standard Instruction Models (Llama-3.1-8B-Instruct). By surgically injecting calculation errors into the model's own Chain-of-Thought (CoT) and measuring its ability to self-correct, we quantify the robustness of its reasoning circuit.

## 📊 Key Findings
* **Reasoning is a Safety Circuit:** The distilled model demonstrated **90% robustness** to error injection, while the baseline model failed **43%** of the time (falling into sycophancy).
* **"Verbalized Dissonance":** In **60%** of recovery cases, the reasoning model explicitly vocalized doubt ("Wait," "Actually," "Hold on"), proving the trace is causally monitored.
* **Structural Capability:** Control experiments show that simple prompting cannot replicate this behavior; the capability is intrinsic to the RL/Distillation training.

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `intervention_experiment.py` | **Main Experiment:** Runs the "Truth Anchor" stress test on 50 GSM8K problems. Implements the "Context Surgery" and "Implication Force" injection protocols. |
| `extended_experiments.py` | **Control Experiments:** Runs the "System 2 Prompt" (Llama-3 skepticism) and "Gaslighting" (Benign variation) stress tests. |
| `analyze_controls.py` | **Analysis Engine:** Processes raw logs to calculate recovery rates, rejection rates, and correlations with CoT complexity. |
| `generate_plots.py` | **Visualization:** Generates the figures (Stacked Bar, Pie Charts, Sensitivity Analysis) used in the report. |
| `data/` | Contains raw JSON logs (`intervention_results.json`, `control_experiments.json`). |

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* A GPU with at least 16GB VRAM (for 8B models in 4-bit)
* `pip install torch transformers bitsandbytes accelerate datasets matplotlib seaborn pandas numpy`

### 1. Run the Main Experiment
Generate the baseline "Truth Anchor" metrics (N=50).
```bash
python intervention_experiment.py
