# ChatMOF: MOF Name Generation and Verification

A streamlined MOF (Metal-Organic Framework) discovery system that uses large language models to generate MOF names and verifies them against the CoreMOF database to find high surface area materials.

## Overview

This framework implements an iterative approach to MOF discovery:
1. **Generate** MOF name candidates using LLMs with structured thinking processes
2. **Verify** candidates by looking up their properties in the CoreMOF database  
3. **Learn** from successful and failed attempts across iterations
4. **Optimize** towards MOFs with surface areas above a specified threshold

## Key Features

- **Structured LLM Reasoning**: Uses thinking processes to improve generation quality
- **Database-Driven Patterns**: Learns from 12,020 real MOFs in the CoreMOF database
- **Robust Parsing**: Handles various LLM output formats with delimiter-based extraction
- **Iterative Learning**: Incorporates all historical attempts (successes and failures)
- **Multi-Provider Support**: Works with OpenAI, Anthropic, and other LLM providers via litellm
- **Comprehensive Logging**: Full experiment tracking with weave integration

## Project Structure

```
chat-mof/
├── cli_generate_verify.py     # Main CLI for MOF generation and verification
├── data/
│   └── coremof.xlsx          # CoreMOF database (12,020 MOFs)
├── src/
│   ├── generation.py         # LLM generation wrapper (litellm + weave)
│   ├── mof_name_generator.py # MOF-specific prompts with thinking processes
│   ├── oracle.py             # Database lookup and evaluation
│   ├── prompt.py             # Base prompt management
│   └── workflow.py           # Optimization workflow (optional)
├── results/                  # Generated experiment results
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation

1. Ensure you have Python 3.8+ installed
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure your LLM credentials according to sde_harness documentation
4. The CoreMOF database is included in `data/coremof.xlsx`

## Usage

### Quick Start

Generate MOF names and verify against database:
```bash
python cli_generate_verify.py --model openai/gpt-4o-mini --target-surface-area 1500 --population-size 15 --n-generations 3
```

### Advanced Usage with Reasoning Models

For better results with reasoning models like o3:
```bash
python cli_generate_verify.py --model openai/o3-2025-04-16 --target-surface-area 3000 --population-size 10 --n-generations 5 --show-all-attempts
```

### Parameters

**Core Parameters:**
- `--model`: LLM model to use (default: "openai/gpt-4o-mini")
- `--target-surface-area`: Surface area threshold in m²/g (default: 2000.0)

**Generation Parameters:**
- `--population-size`: MOF names generated per iteration (default: 10)
- `--n-generations`: Number of iterations (default: 5)

**Learning Parameters:**
- `--use-iterative-learning`: Learn from previous attempts (default: True)
- `--database-examples`: Real MOF examples to show in prompts (default: 20)

**Display Options:**
- `--show-all-attempts`: Show all generated names, not just successful ones
- `--project-name`: Weave project name for logging (default: "ChatMOF-Generate-Verify")

## How It Works

### 1. MOF Name Generation
- Uses structured prompts with thinking processes for reasoning models
- Focuses on CSD codes (6-letter identifiers: VEJYIT, ETAXUT, PENNON)
- Includes numbered variants (VEJYIT01, ETAXUT02) and literature codes
- Avoids systematic names (UiO-66, ZIF-8) which are rare in this database

### 2. Database Verification
- Looks up generated names in CoreMOF database (12,020 MOFs)
- Extracts surface area and other properties for found MOFs
- Identifies MOFs above the specified threshold

### 3. Iterative Learning
- Shows ALL successful MOFs found in previous iterations
- Shows ALL failed attempts to avoid repetition
- Uses database examples to guide realistic name generation
- Employs structured thinking processes for better reasoning

### 4. Robust Parsing
- Uses "BELOW ARE GENERATED MOFS:" delimiter to separate thinking from names
- Filters out markdown formatting (```) and common non-MOF text
- Handles various LLM output styles reliably

## Example Results

Successful discoveries include:
- **PENNON**: 3,777 m²/g (Co-based, 73% void fraction)
- **ETAXUT**: 2,630 m²/g (Zn-based, 72% void fraction)  
- **VEJYIT**: 2,140 m²/g (Zn-based, 67% void fraction)
- **XIGWUF**: 6,931 m²/g (discovered in other runs)

Typical success rates:
- **First generation**: 8-25% success rate (finding valid high-SA MOFs)
- **Overall performance**: 8-15% across multiple generations
- **Database hit rate**: 10-30% (finding any MOF in database)

## Output Files

Results are saved as JSON files with naming pattern:
```
mof_generation_verification_{model}_{target_surface_area}sa.json
```

Contains:
- Generated MOF names and verification results
- Success statistics per generation
- Detailed properties of discovered MOFs
- Complete history of all attempts

## Database Information

Uses the CoreMOF database with:
- **12,020 unique MOFs** with calculated properties
- **Surface areas**: 0-8,318 m²/g range
- **Naming patterns**: 86.4% CSD codes, 9% numbered variants, 3.3% literature codes
- **Key columns**: name, "Accessible Surface Area (m^2/g)", "Metal type", "void fraction"

## Integration Notes

This is a focused, streamlined version of ChatMOF that:
- Removes LangChain dependencies for simpler deployment
- Uses direct database lookup instead of complex search
- Focuses on iterative MOF name generation rather than general chat
- Integrates with the sde_harness framework for LLM management

## Troubleshooting

**Low success rates?**
- Try reasoning models (o3) which perform better with thinking processes
- Increase `--database-examples` to provide more context
- Use lower `--target-surface-area` thresholds initially

**Parsing issues?**
- The system automatically handles various LLM output formats
- Check that models are following the "BELOW ARE GENERATED MOFS:" delimiter

**Database not found?**
- Ensure `data/coremof.xlsx` exists in the project directory
- File should contain 12,020 rows with MOF properties