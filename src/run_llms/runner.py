from abc import ABC, abstractmethod
from tqdm import tqdm
import pandas as pd
import json

class LLMRunner(ABC):
    def __init__(self, temperature, save_every, model_id):
        self.temperature = temperature
        self.save_every = save_every
        self.model_id = model_id
        self.system_message =  """You are a decision-making assistant. You'll receive a message containing three sections of a message: ##context, ##question and ##options with exactly three options formatted as follows:

Option 0: <option text>
Option 1: <option text>
Option 2: <option text>

Your task is to select one of these options based on the given situation (context) and output only the chosen option’s number and text. Do not provide any explanation or reasoning for your choice.
"""
        self.user_message_template = """
##context
__context__
##question
__question__
##options
Option 0: __option0__
Option 1: __option1__
Option 2: __option2__
"""

    def read_excel(self, path, sheet_name):
        return pd.read_excel(path, sheet_name=sheet_name, engine='openpyxl')

    def process_excel(self, df):
        df = df.dropna(subset=['answer_info'])

        def convert_text_to_json(texto):
            try:
                return json.loads(str(texto).replace("'", '"'))
            except json.JSONDecodeError:
                print(f"Error al convertir a JSON: {texto}")
                return None

        df['answer_info'] = df['answer_info'].apply(convert_text_to_json)
        return df.dropna(subset=['answer_info'])

    def create_user_message(self, context, question, answer_info):
        return self.user_message_template.replace('__context__', context)\
            .replace('__question__', question)\
            .replace('__option0__', answer_info['ans0'])\
            .replace('__option1__', answer_info['ans1'])\
            .replace('__option2__', answer_info['ans2'])

    def run_llm(self, client, df, output_path):
        df['result'] = None
        df['tokens'] = None

        for i, row in tqdm(df.iterrows(), total=len(df), disable=False):
            try:
                model_answer = self.run_one_prompt(client, row)
                df.at[i, 'result'] = model_answer
            except Exception as e:
                print(f"Error on line {i}: {e}")

            if (i + 1) % self.save_every == 0:
                df.to_csv(output_path, index=False)

        df.to_csv(output_path, index=False)

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def run_one_prompt(self, row):
        pass
