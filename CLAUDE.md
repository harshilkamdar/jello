# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Jello** is a research framework for mechanistic interpretability of jailbreak/refusal behavior in open-source MoE chat models (e.g., GPT-OSS-20B). The goal is to understand and control safety mechanisms through activation analysis, circuit identification, and lightweight weight edits.

### Research Goals
- Identify low-rank refusal directions and test steering/ablation approaches
- Analyze attention head behavior during jailbreaks (which heads carry refusal vs compliance signals)
- Study MoE routing patterns: how gate logits correlate with refusal/compliance
- Compare prompt-only vs weight-adjacent jailbreak methods
- Develop safety-helpfulness Pareto analysis for interventions

## Development Commands

This project uses UV for Python package management and Make for common tasks:

### Environment Setup
- `make install` - Install environment and pre-commit hooks (required first step)
- `uv sync` - Install dependencies from lock file

### Code Quality
- `make check` - Run all quality checks (linting, type checking, dependency checks)
- `uv run pre-commit run -a` - Run pre-commit hooks on all files
- `uv run mypy` - Type checking with mypy
- `uv run deptry src` - Check for obsolete dependencies

### Testing
- `make test` - Run tests with pytest (includes doctests)
- `uv run python -m pytest --doctest-modules` - Run tests manually

### Building
- `make build` - Build wheel file
- `make clean-build` - Clean build artifacts

### CLI Usage
The main CLI entry point is `jello` (defined in `[project.scripts]`):
- `uv run jello chat --prompt "text" --model "openai/gpt-oss-20b"` - Quick single-turn chat
- `uv run jello runfile --prompts file.jsonl --attack system_leak` - Batch process with attack scenarios
- `uv run jello inspect --prompt "text" --attn --moe` - Analyze model internals with attention and MoE routing

## Architecture Overview

### Core Components

**Model Interface (`src/jello/llm/runner.py`)**
- `GPTOSS` dataclass: Main model wrapper for GPT-OSS and similar models
- `load_gptoss()`: Model loader with dtype/device configuration for Runpod GPU setups
- `.chat()`: Standard chat interface with HF chat template support
- `.forward_inspect()`: Forward pass capturing hidden states, attention maps, and MoE routing for analysis

**Analysis Tools (`src/jello/analysis/introspect.py`)**
- `MoESummary`: Data structure for MoE routing analysis (expert selection patterns)
- `summarize_router_logits()`: Extract top-k expert routing per layer for jailbreak analysis
- `layer_activation_norms()`: L2 norms across layers and sequence positions
- `attention_head_entropies()`: Attention distribution entropy for identifying anomalous patterns

**Attention Probing (`src/jello/probes/attention.py`)**
- `AttentionProbeResult`: Structured attention analysis data
- `collect_attentions()`: Forward pass to capture attention maps for circuit analysis
- `head_distraction_index()`: Quantify attention shift between instruction vs adversarial injection spans
- Visualization utilities for attention heatmaps and head-level distraction analysis

**CLI Interface (`src/jello/lab.py`)**
- `chat`: Interactive testing for model behavior verification
- `runfile`: Batch processing with attack scenarios (baseline, system_leak) for refusal analysis
- `inspect`: Model internals capture for mechanistic analysis (outputs to `.runs/inspect/`)

**Configuration (`src/jello/config.py`)**
- Pydantic settings for model parameters optimized for research workflows

### Key Dependencies
- **Core ML**: torch, transformers, accelerate (HF reference implementation)
- **Data**: datasets, jsonlines, pandas (for prompt processing and results)
- **Analysis**: matplotlib, numpy (for activation analysis and visualization)
- **Interface**: typer, rich (for CLI and readable outputs)

### Research Workflow

1. **Data Collection**: Use `runfile` to batch process refusal/jailbreak prompt sets
2. **Activation Analysis**: Use `inspect` to capture internals for specific prompts
3. **Circuit Identification**: Use attention probes to identify refusal-related heads/paths
4. **MoE Analysis**: Examine routing patterns during refusal vs compliance
5. **Intervention Testing**: Apply steering vectors or ablations (to be implemented)

### Output Structure
- `.runs/run.jsonl` - Batch processing results with attack outcomes
- `.runs/inspect/` - Model inspection outputs:
  - `activation_norms.pt` - Layer-wise activation patterns
  - `attention_entropies.pt` - Head-wise attention distributions
  - `moe_topk.json` - Expert routing summaries
  - `summary.json` - Consolidated metrics

### Planned Extensions
- **Refusal direction analysis**: Low-rank subspace identification and projection tools
- **Causal interventions**: Activation patching and head ablation utilities
- **Steering mechanisms**: Rank-1 orthogonalization and LoRA-style steering modules
- **Safety-helpfulness evaluation**: Pareto analysis framework for intervention effects