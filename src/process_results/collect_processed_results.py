import pandas as pd
from pathlib import Path
import argparse

def load_and_merge(base_df, filepath, colname, filter_bbq=True):
    if not Path(filepath).exists():
        return base_df
    df = pd.read_excel(filepath)
    if filter_bbq and 'bbq' in df.columns:
        df = df[df['bbq'] == False]
    df = df[['context', 'question', 'answer', 'label', 'probab_label']]
    df.rename(columns={'probab_label': colname}, inplace=True)
    return pd.merge(base_df, df, on=['context', 'question', 'answer', 'label'], how='outer')

def collect_results(input_dir, language):
    temps = ['01', '025', '05', '075', '1']
    bias_types = ['racismo', 'clasismo', 'genero', 'xenofobia']

    models = {
        "claude":        lambda bt, t: f"claude/{bt}_claude_{t}_processed.xlsx",
        "deepseek":      lambda bt, t: f"deepseek/results_DeepSeekR1_7B_{bt}_T{t}.xlsx",
        "gemini":        lambda bt, t: f"gemini/{bt}_gemini_{t}_processed.xlsx",
        "gpt":           lambda bt, t: f"gpt/{bt}_gpt_temp_{t}_processed.xlsx",
        "llama":         lambda bt, t: f"llama/{bt}_llama_31_8B_temp_{t}_processed.xlsx",
        "llama_unc":     lambda bt, t: f"llama_uncensored/{bt}_llama_uncensored_31_8B_temp_{t}_processed.xlsx",
    }

    colnames = {
        "claude":        lambda t: f'results_Claude_T{t}',
        "deepseek":      lambda t: f'results_DeepSeekR1_7B_T{t}',
        "gemini":        lambda t: f'results_Gemini_T{t}',
        "gpt":           lambda t: f'results_gpt4omini_temp_{t}_num',
        "llama":         lambda t: f'results_llama_318B_temp_{t}_num',
        "llama_unc":     lambda t: f'results_llama_318B_uncensored_temp_{t}_num',
    }

    all_dfs = []

    for bt in bias_types:
        # Claude base
        base_fp = Path(input_dir) / f"claude/{bt}_claude_01_processed.xlsx"
        df = pd.read_excel(base_fp)
        df.rename(columns={'probab_label': colnames["claude"]("01")}, inplace=True)

        # Agregar modelos y temperaturas
        for model_key, path_fn in models.items():
            for t in (temps if model_key != "claude" else temps[1:]):
                fp = Path(input_dir) / path_fn(bt, t)
                df = load_and_merge(df, fp, colnames[model_key](t))

        df['tipo'] = bt
        if 'id_prompt' in df.columns:
            df.rename(columns={'id_prompt': 'prompt_id'}, inplace=True)
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    out_path_all = f'aggregated_results_temps_{language}.xlsx'

    with pd.ExcelWriter(out_path_all) as writer:
        df_all.to_excel(writer, index=False, sheet_name='All')
        for bt, df in zip(bias_types, all_dfs):
            df.to_excel(writer, index=False, sheet_name=bt.capitalize())

    # Subset para temp 075
    models_075 = {
        colnames["gpt"]("075"): "GPT-4o mini",
        colnames["llama"]("075"): "Llama 3.1 Instruct",
        colnames["llama_unc"]("075"): "Llama 3.1 Uncensored",
        colnames["deepseek"]("075"): "DeepSeek R1",
        colnames["gemini"]("075"): "Gemini2.0 Flash",
        colnames["claude"]("075"): "Claude 3.5 Haiku",
    }

    all_075 = []
    for df in all_dfs:
        cols = ['tipo', 'prompt_id', 'question_polarity', 'context_condition', 'label', 'target', 'other'] + list(models_075.keys())
        if all(c in df.columns for c in cols):
            df_075 = df[cols].copy()
            df_075.rename(columns=models_075, inplace=True)
            all_075.append(df_075)

    df_075_all = pd.concat(all_075, ignore_index=True)
    out_path_075 = f'aggregated_results_temps_075_{language}.xlsx'

    with pd.ExcelWriter(out_path_075) as writer:
        df_075_all.to_excel(writer, index=False, sheet_name='All')
        for bt, df in zip(bias_types, all_075):
            df.to_excel(writer, index=False, sheet_name=bt.capitalize())

    return df_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--language', type=str, required=True, choices=['es', 'en'])
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()

    df_all = collect_results(args.input_dir, args.language)

    output_path = args.output_path or f'../../results_prompts/Resultados_agregados_Temperaturas_{args.language}.xlsx'
    if not df_all.empty:
        df_all.to_excel(output_path, index=False)
        print(f"Aggregated results saved to {output_path}")
    else:
        print("No results to save.")
