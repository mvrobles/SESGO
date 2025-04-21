from dotenv import load_dotenv
from google.genai import types
from google import genai
import os

from runner import LLMRunner

class GeminiRunner(LLMRunner):
    def __init__(self, temperature, save_every, model_id):
        super().__init__(temperature, save_every, model_id)

    def connect(self):
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        client = genai.Client(api_key = api_key)
        return client

    def run_one_prompt(self, client, row):
        user_message = self.create_user_message(row.context, row.question, row.answer_info)

        response = client.models.generate_content(
            model=self.model_id,
            config=types.GenerateContentConfig(
                system_instruction=self.system_message,
                temperature=self.temperature),
            contents=user_message
        )

        return response.text