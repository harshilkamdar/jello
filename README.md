# jello

[![Release](https://img.shields.io/github/v/release/harshilkamdar/jello)](https://img.shields.io/github/v/release/harshilkamdar/jello)
[![Build status](https://img.shields.io/github/actions/workflow/status/harshilkamdar/jello/main.yml?branch=main)](https://github.com/harshilkamdar/jello/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/harshilkamdar/jello/branch/main/graph/badge.svg)](https://codecov.io/gh/harshilkamdar/jello)
[![Commit activity](https://img.shields.io/github/commit-activity/m/harshilkamdar/jello)](https://img.shields.io/github/commit-activity/m/harshilkamdar/jello)
[![License](https://img.shields.io/github/license/harshilkamdar/jello)](https://img.shields.io/github/license/harshilkamdar/jello)

A research framework for mechanistic interpretability of jailbreak and refusal behavior in open-source MoE chat models. Jello provides tools for activation analysis, circuit identification, and controlled interventions to understand safety mechanisms in large language models.

## Features

- **Model Analysis**: Deep inspection of GPT-OSS and similar models with activation capture
- **MoE Routing Analysis**: Expert selection pattern analysis during refusal/compliance
- **Attention Probing**: Head-level analysis of attention shifts during jailbreaks
- **Batch Processing**: Systematic evaluation of refusal patterns across prompt sets
- **Visualization Tools**: Clean outputs for attention maps, routing patterns, and activation flows

## Quick Start

### Installation

```bash
git clone https://github.com/harshilkamdar/jello.git
cd jello
make install
```

### Basic Usage

```bash
# Quick chat test
uv run jello chat --prompt "How do I make a bomb?" --model "openai/gpt-oss-20b"

# Batch analysis with jailbreak attempts
uv run jello runfile --prompts prompts.jsonl --attack system_leak

# Deep model inspection
uv run jello inspect --prompt "Harmful request" --attn --moe
```

## Research Applications

- **Refusal Subspace Analysis**: Identify low-rank directions encoding refusal behavior
- **Circuit Identification**: Locate attention heads and pathways carrying safety signals
- **Intervention Testing**: Test steering vectors and ablations on safety vs helpfulness
- **Attack Analysis**: Compare prompt-only vs gradient-based jailbreak methods

- **Github repository**: <https://github.com/harshilkamdar/jello/>
- **Documentation** <https://harshilkamdar.github.io/jello/>

## Getting started with your project

### 1. Create a New Repository

First, create a repository on GitHub with the same name as this project, and then run the following commands:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:harshilkamdar/jello.git
git push -u origin main
```

### 2. Set Up Your Development Environment

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

### 3. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 4. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Releasing a new version



---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
