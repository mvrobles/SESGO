# Packages
import os
import pandas as pd
from glob import glob
import argparse

def collect_results(input_dir, language):
    """
    Collects all .xlsx files from the input directory (recursively), extracts metadata from filenames,
    and concatenates them into a single DataFrame with additional columns for bias type, temperature, model, and language.

    Args:
        input_dir (str): Path to the directory containing the result .xlsx files.
        language (str): Language code ("es" or "en").

    Returns:
        pd.DataFrame: Concatenated DataFrame with all results and metadata columns.
    """
    temps = ['01', '025', '05', '075', '1']
    bias_types = ['racismo', 'clasismo', 'genero', 'xenofobia']

    dfs = {}
    for bt in bias_types:
        # Add in the loop the models you are using

        dfs[f'results_{bt}'] = pd.read_excel(input_dir + f'/claude/{bt}_claude_01_processed.xlsx')
        dfs[f'results_{bt}'].rename(columns={'probab_label': 'results_Claude_T01'}, inplace=True)

        for i in range(len(temps)-1):
            t = temps[i+1]
            results_temp = pd.read_excel(input_dir + f'claude/{bt}_claude_{t}_processed.xlsx')
            results_temp = results_temp[results_temp['bbq'] == False]
            results_temp.rename(columns={'probab_label': f'results_Claude_T{t}'}, inplace=True)
            dfs[f'results_{bt}'] = pd.merge(dfs[f'results_{bt}'], results_temp[['context', 'question', 'answer', 'label', f'results_Claude_T{t}']], on=['context', 'question', 'answer', 'label'], how='outer')

        for i in range(len(temps)):
            t = temps[i]
            results_temp = pd.read_excel(input_dir + f'deepseek/results_DeepSeekR1_7B_{bt}_T{t}.xlsx')
            results_temp = results_temp[results_temp['bbq'] == False]
            results_temp.rename(columns={'probab_label': f'results_DeepSeekR1_7B_T{t}'}, inplace=True)
            dfs[f'results_{bt}'] = pd.merge(dfs[f'results_{bt}'], results_temp[['context', 'question', 'answer', 'label', f'results_DeepSeekR1_7B_T{t}']], on=['context', 'question', 'answer', 'label'], how='outer')

        for i in range(len(temps)):
            t = temps[i]
            results_temp = pd.read_excel(input_dir + f'gemini/{bt}_gemini_{t}_processed.xlsx')
            results_temp = results_temp[results_temp['bbq'] == False]
            results_temp.rename(columns={'probab_label': f'results_Gemini_T{t}'}, inplace=True)
            dfs[f'results_{bt}'] = pd.merge(dfs[f'results_{bt}'], results_temp[['context', 'question', 'answer', 'label', f'results_Gemini_T{t}']], on=['context', 'question', 'answer', 'label'], how='outer')

        for i in range(len(temps)):
            t = temps[i]
            results_temp = pd.read_excel(input_dir + f'gpt/{bt}_gpt_temp_{t}_processed.xlsx')
            results_temp = results_temp[results_temp['bbq'] == False]
            results_temp.rename(columns={'probab_label': f'results_gpt4omini_temp_{t}_num'}, inplace=True)
            dfs[f'results_{bt}'] = pd.merge(dfs[f'results_{bt}'], results_temp[['context', 'question', 'answer', 'label', f'results_gpt4omini_temp_{t}_num']], on=['context', 'question', 'answer', 'label'], how='outer')

        for i in range(len(temps)):
            t = temps[i]
            results_temp = pd.read_excel(input_dir + f'llama/{bt}_llama_31_8B_temp_{t}_processed.xlsx')
            results_temp = results_temp[results_temp['bbq'] == False] 
            results_temp.rename(columns={'probab_label': f'results_llama_318B_temp_{t}_num'}, inplace=True)
            dfs[f'results_{bt}'] = pd.merge(dfs[f'results_{bt}'], results_temp[['context', 'question', 'answer', 'label', f'results_llama_318B_temp_{t}_num']], on=['context', 'question', 'answer', 'label'], how='outer')

        for i in range(len(temps)):
            t = temps[i]
            results_temp = pd.read_excel(input_dir + f'llama_uncensored/{bt}_llama_uncensored_31_8B_temp_{t}_processed.xlsx') 
            results_temp = results_temp[results_temp['bbq'] == False] 
            results_temp.rename(columns={'probab_label': f'results_llama_318B_uncensored_temp_{t}_num'}, inplace=True)
            dfs[f'results_{bt}'] = pd.merge(dfs[f'results_{bt}'], results_temp[['context', 'question', 'answer', 'label', f'results_llama_318B_uncensored_temp_{t}_num']], on=['context', 'question', 'answer', 'label'], how='outer', validate='1:1')

        dfs[f'results_{bt}']['tipo'] = bt
        dfs[f'results_{bt}'].rename(columns={'id_prompt': 'prompt_id'}, inplace=True)
    
    df_all = pd.concat([dfs[f'results_{bt}'] for bt in bias_types], ignore_index=True)
    excel_file = f'aggregated_results_temps_{language}.xlsx'

    with pd.ExcelWriter(excel_file) as writer:
        df_all.to_excel(writer, index=False, sheet_name='All')
        for bt in bias_types: dfs[f'results_{bt}'].to_excel(writer, index=False, sheet_name=f'{bt.capitalize()}')

    results_75 = {}
    for bt in bias_types:

        results_75[bt] = dfs[f'results_{bt}'][['tipo', 'prompt_id', 'question_polarity', 'context_condition', 'label',
       'target', 'other','results_gpt4omini_temp_075_num', 'results_llama_318B_temp_075_num',
       'results_llama_318B_uncensored_temp_075_num', 'results_DeepSeekR1_7B_T075', 
       'results_Gemini_T075', 'results_Claude_T075']].copy()

        results_75[bt].rename(columns={'results_gpt4omini_temp_075_num': 'GPT-4o mini',
                            'results_llama_318B_temp_075_num': 'Llama 3.1 Instruct',
                            'results_llama_318B_uncensored_temp_075_num': 'Llama 3.1 Uncensored',
                            'results_DeepSeekR1_7B_T075': 'DeepSeek R1',
                            'results_Gemini_T075': 'Gemini2.0 Flash',
                            'results_Claude_T075': 'Claude 3.5 Haiku'}, inplace=True)
        
    df_all_75 = pd.concat([results_75[bt] for bt in bias_types], ignore_index=True)
    excel_file_75 = f'aggregated_results_temps_075_{language}.xlsx'
    
    with pd.ExcelWriter(excel_file_75) as writer:
        df_all_75.to_excel(writer, index=False, sheet_name='All')
        for bt in bias_types:
            results_75[bt].to_excel(writer, index=False, sheet_name=f'{bt.capitalize()}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect processed results and aggregate into a single Excel file.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing processed .xlsx files')
    parser.add_argument('--language', type=str, required=True, choices=['es', 'en'], help='Language code: es or en')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save the aggregated Excel file')
    args = parser.parse_args()

    # Set output path with language in the filename if not provided
    if args.output_path is None:
        output_path = f'../../results_prompts/Resultados_agregados_Temperaturas_{args.language}.xlsx'
    else:
        output_path = args.output_path

    df_all = collect_results(args.input_dir, args.language)
    if not df_all.empty:
        df_all.to_excel(output_path, index=False)
        print(f"Aggregated results saved to {output_path}")
    else:
        print("No .xlsx files found in the specified directory.")


