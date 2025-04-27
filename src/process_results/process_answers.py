from glob import glob
import pandas as pd
import argparse
import json

def select_answer(prompt):
  answer_info = json.loads(prompt.answer_info.replace("'", '"'))
  answer = prompt.result

  if pd.isna(answer):
    return None

  for key, value in answer_info.items():
    num = key.replace('ans', '')
    if answer in [f"Option {num}: {value}", f"{num}: {value}", f"{num}"] or value in answer:
      return num
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LLM on Excel data')
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)

    path = parser.parse_args().input_path
    output_path = parser.parse_args().output_path

    df_temp = pd.read_csv(path)
    df_temp['probab_label'] = df_temp.apply(lambda row: select_answer(row), axis = 1)
    df_temp.to_excel(output_path, index = None)