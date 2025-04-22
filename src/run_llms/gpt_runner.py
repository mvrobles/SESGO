from openai import AzureOpenAI
from dotenv import load_dotenv
import os

from runner import LLMRunner

class GPTRunner(LLMRunner):
    def __init__(self, temperature, save_every, model_id):
        super().__init__(temperature, save_every, model_id)

    def connect(self):
        load_dotenv()
        api_key = os.getenv('GPT_API_KEY')
        endpoint = os.getenv('GPT_ENDPOINT')
        client = AzureOpenAI(
            azure_endpoint = endpoint,
            api_key = api_key,
            api_version="2024-10-21"
            )
        return client

    def run_one_prompt(self, client, row):
        user_message = self.create_user_message(row.context, row.question, row.answer_info)

        messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_message}
            ]
        response = client.chat.completions.create(
        model = self.model_id,
        messages = messages,
        temperature=self.temperature 
        )

        return response.choices[0].message.content
