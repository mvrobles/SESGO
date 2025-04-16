from openai import AzureOpenAI
from dotenv import load_dotenv
import os

from runner import LLMRunner

class DeepseekRunner(LLMRunner):
    def __init__(self, temperature, save_every):
        super().__init__(temperature, save_every)

    def load_model(self):
        load_dotenv()
        api_key = os.getenv('API_KEY')
        client = AzureOpenAI(
            azure_endpoint = os.getenv('ENDPOINT'),
            api_key = os.getenv('API_KEY', api_key),
            api_version="2024-10-21"
            )
        return client

    def run_one_prompt(self, client, row, temperature):
        user_message = self.create_user_message(row.context, row.question, row.answer_info)

        messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_message}
            ]
        response = client.chat.completions.create(
        model = 'gpt',
        messages = messages,
        temperature=temperature 
        )
        total_tokens = response.usage.total_tokens

        return response.choices[0].message.content, total_tokens
