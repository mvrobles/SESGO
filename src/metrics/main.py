import os
import pandas as pd
from temperatures import (
    calculate_metrics,
    plot_results,
    process_temperature_metrics,
    plot_bias_vs_temperature
)
from language_metrics import load_and_process_metrics, get_merged_bias_scores, plot_bias_scores
from metrics import get_type_metrics

def main():
    """
    Main function to process and analyze model results and metrics.

    This function sets up relative paths for input and output directories, loads result files,
    computes bias metrics for different models and temperatures, and generates summary tables and plots.
    It handles both English and Spanish datasets and produces outputs for ambiguous and disambiguated scenarios.

    Steps performed:
        1. Define and create necessary directories for results, tables, and figures.
        2. Load aggregated results and temperature-specific results from Excel files.
        3. Calculate bias metrics for a list of models and save the results as Excel tables.
        4. Generate and save plots for ambiguous and disambiguated scenarios.
        5. Process temperature metrics and plot bias as a function of temperature.
        6. Load and process multilingual metrics (English and Spanish), merge bias scores, and plot multilingual bias evaluations.

    Note:
        - All output folders are created outside the 'src' directory, relative to the project root.
        - The function assumes the existence of specific input files in the 'results_prompts' directory.

    Returns:
        None
    """
        
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    path = os.path.join(base_dir, 'results_prompts')
    output_path = os.path.join(base_dir, 'results', 'tables')
    output_path_temp = os.path.join(base_dir, 'results', 'tables', 'fullTemperatures')
    output_graph = os.path.join(base_dir, 'results', 'figures')
    output_path_multi = os.path.join(base_dir, 'results', 'tables', 'multilingual')

    # Create output directories if they don't exist
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path_temp, exist_ok=True)
    os.makedirs(output_graph, exist_ok=True)
    os.makedirs(output_path_multi, exist_ok=True)

    models = ['GPT-4o mini', 'Llama 3.1 Instruct', 'Llama 3.1 Uncensored',
              'DeepSeek R1', 'Gemini 2.0 Flash', 'Claude 3.5 Haiku']

    models_temp = [
        "gpt4omini_01", "gpt4omini_025", "gpt4omini_05", "gpt4omini_075", "gpt4omini_1",
        "llama_01", "llama_025", "llama_05", "llama_075", "llama_1",
        "llama_uncensored_01", "llama_uncensored_025", "llama_uncensored_05", "llama_uncensored_075", "llama_uncensored_1",
        "DeepSeekR1_7B_T01", "DeepSeekR1_7B_T025", "DeepSeekR1_7B_T05", "DeepSeekR1_7B_T075", "DeepSeekR1_7B_T1",
        "Gemini_T01", "Gemini_T025", "Gemini_T05", "Gemini_T075", "Gemini_T1",
        "Claude_T01", "Claude_T025", "Claude_T05", "Claude_T075", "Claude_T1"
    ]
    
    #Please change the paths according to your datasets
    df_main = pd.read_excel(os.path.join(path, 'Resultados_agregados_vf_T075.xlsx'))
    df_temperature = pd.read_excel(os.path.join(path, 'Resultados_agregados_Temperaturas.xlsx'))
    df_english = pd.read_excel(f'{path}/Resultados inglés.xlsx', sheet_name='Sheet1')
    df_spanish = pd.read_excel(f'{path}/Resultados inglés.xlsx', sheet_name='Sheet2')

    df_ambig, df_disamb = calculate_metrics(df_main, models)
    df_disamb_075,df_ambig_075 = get_type_metrics(df_main, models)
    df_ambig_075=df_ambig_075.pivot(index='type', columns='model', values='bias_score')[models]
    df_disamb_075=df_disamb_075.pivot(index='type', columns='model', values='bias_score')[models]

    df_ambig_075.to_excel(os.path.join(output_path, "metrics_ambiguous.xlsx"), index=True)
    df_disamb_075.to_excel(os.path.join(output_path, "metrics_disambiguated.xlsx"), index=True)

    eps = {
        'Ambiguous': {'GPT-4o mini': 0.01, 'Llama 3.1 Instruct': 0, 'Llama 3.1 Uncensored': -0.0,
                      'DeepSeek R1': -0.017, 'Gemini 2.0 Flash': -0.01, 'Claude 3.5 Haiku': -0.01},
        'Disambiguated': {'GPT-4o mini': 0.01, 'Llama 3.1 Instruct': 0, 'Llama 3.1 Uncensored': -0.0,
                         'DeepSeek R1': -0.01, 'Gemini 2.0 Flash': -0.01, 'Claude 3.5 Haiku': -0.01}
    }

    plot_results(df_ambig, 'Ambiguous', eps,output_graph)
    plot_results(df_disamb, 'Disambiguated', eps,output_graph)
    df_temp_plot = process_temperature_metrics(df_temperature, models_temp, output_path_temp)

    model_name_map = {
        "llama_uncensored": "Llama 3.1 Uncensored",
        "llama": "Llama 3.1 Instruct",
        "gpt4omini": "GPT-4o mini",
        "DeepSeekR1_7B": "DeepSeek R1",
        "Gemini": "Gemini 2.0 Flash",
        "Claude": "Claude 3.5 Haiku"
    }

    def extract_base_model(model_str):
        for key in model_name_map:
            if key in model_str:
                return model_name_map[key]
        return model_str

    df_temp_plot["base_model"] = df_temp_plot["model"].apply(extract_base_model)
    plot_bias_vs_temperature(df_temp_plot, output_graph)

    df_by_type = load_and_process_metrics(df_english, df_spanish, models,output_path_multi)
    df_type_disamb, df_type_ambig = get_type_metrics(df_main, models)

    df_en_ambig = df_by_type['ambig']['en']
    df_es_ambig = df_by_type['ambig']['es']
    df_en_disamb = df_by_type['disamb']['en']
    df_es_disamb = df_by_type['disamb']['es']

    en_amb = df_en_ambig.query("type == 'xpooled'")
    es_amb = df_es_ambig.query("type == 'xpooled'")
    es_full_amb = df_type_ambig.query("type == 'xpooled'")

    en_disamb = df_en_disamb.query("type == 'xpooled'")
    es_disamb = df_es_disamb.query("type == 'xpooled'")
    es_full_disamb = df_type_disamb.query("type == 'xpooled'")

    amb_scores = get_merged_bias_scores(en_amb, es_amb, es_full_amb, bias_label_prefix='amb')
    disam_scores=get_merged_bias_scores(en_disamb,es_disamb, es_full_disamb, bias_label_prefix='disam')

    plot_bias_scores(amb_scores, bias_prefix='amb', title='Multilingual Bias Evaluation in Ambiguous Scenarios',output_path=output_graph)
    plot_bias_scores(disam_scores, bias_prefix='disam', title='Multilingual Bias Evaluation in Disambiguous Scenarios',output_path=output_graph)

if __name__ == "__main__":
    main()
