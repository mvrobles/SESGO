# SESGO (Spanish Evaluation of Stereotypical Generative Outputs)

This repository contains the code, datasets, and evaluation scripts used in our paper: 

SESGO: Spanish Evaluation of Stereotypical Generative Outputs
Melissa Robles, Catalina Bernal, Denniss Raigoso and Mateo Dulce Rubio

## Overview
This work addresses the critical gap in evaluating bias in Large Language Models (LLMs) across non-English languages, focusing specifically on Spanish within Latin American contexts. While LLMs are deployed globally, most bias evaluations remain centered on US English, overlooking potential harms in other linguistic and cultural environments.

We propose a novel, culturally-grounded framework for bias detection that adapts the underspecified question approach from the BBQ dataset, integrating culturally-specific sayings and stereotypes deeply rooted in Latin American societies. Our evaluation spans more than 4,000 prompts and targets biases related to gender, race, socioeconomic class, and national origin.

Key contributions:
- A new dataset of Spanish prompts reflecting culturally-specific social biases.
- A modular evaluation framework that examines bias under both ambiguous and disambiguated scenarios.
- A proposed metric that balances accuracy and bias alignment.
- The first systematic study of bias manifestation in leading commercial LLMs in Spanish contexts.

## Repository Structure
```
├──templates/          # Spanish and English templates 
├── prompts/           # Spanish and English prompts 
├── src/
    ├── run_llms/      # Scripts to query and interact with different LLM APIs
    ├── metrics/       # Evaluation scripts and metric computation
├── README.md          # This file
├── pyproject.toml     # Project dependencies and configuration
└── pyproject.toml     # List of Python packages for easy installation
```

## Getting started
#### 1. Clone the repository:
```
git clone https://github.com/mvrobles/SESGO.git
cd SESGO
```
#### 2. Install the required packages:

We recommend using [uv](https://github.com/astral-sh/uv) for faster and more reliable dependency management.
```
uv sync
```

#### 3. Set up API keys for the different model providers:

Create a `.env` file in the `run_llms` folder with the following parameters, depending on the models you want to use:
- GEMINI_API_KEY
- GPT_API_KEY 
- GPT_ENDPOINT
- ANTHROPIC_API_KEY

#### 4. Run LLMs

Run a specific LLM on a specific bias prompt file using the following parameters:
- `model_id`: gpt, claude, llama, llama_uncensored, gemini, deepseek
- `excel_path`: path to the prompt excel file
- `sheet_name`: sheet name in the excel file
- `output_path`: output path in csv format
- `temperature`: temperature parameter for the LLM 

```
uv run python run_llm.py --model_id <model_id> --excel_path <path.xlsx> --sheet_name <sheet_name> --output_path <output_path.csv> --temperature <temperature>
```

#### 5. Process and validate the results

```
uv run python compute_metrics.py --input_path <input_path.csv> --output_path <output_path.xlsx> 
```
#### 6. Run metrics

**Prerequisites:**

- Required packages installed (e.g., `pandas`, `openpyxl`, `matplotlib`, etc.)  
- Dataset Excel files located in the appropriate folder structure (see paths below)  

---

**How to Run**

1. Check and update paths in the script before running:

   - `path` — folder containing input Excel files with raw results  
   - `output_path` — folder where output tables will be saved  
   - `output_path_temp` — folder where temperature-related tables will be saved  
   - `output_graph` — folder where figures will be saved  
   - `output_path_multi` — folder for multilingual output tables  

   Update these paths to match your local file system.

2. Input files expected:

   - `Resultados_agregados_vf_T075.xlsx`  
   - `Resultados_agregados_Temperaturas.xlsx`  
   - `Resultados inglés.xlsx` (with sheets 'Sheet1' and 'Sheet2')  

3. Run the script:

   ```bash
   src/metrics/main.py
### What the script does:

- Loads data from the Excel files  
- Calculates bias metrics for specified models  
- Saves bias metric tables (`metrics_ambiguous.xlsx`, `metrics_disambiguated.xlsx`) to the output folder  
- Generates and saves plots for bias and temperature effects  
- Performs multilingual bias evaluations and saves results  

---

### Notes:

- Make sure you have all required dependencies installed. You can install missing Python packages with:

  ```bash
  pip install pandas openpyxl matplotlib seaborn

