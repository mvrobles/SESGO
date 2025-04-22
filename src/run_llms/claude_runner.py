from dotenv import load_dotenv
import anthropic
import os

from runner import LLMRunner

class ClaudeRunner(LLMRunner):
    def __init__(self, temperature, save_every, model_id):
        super().__init__(temperature, save_every, model_id)

    def connect(self):
        load_dotenv()
        api_key = os.getenv('ANTHROPIC_API_KEY')
        client = anthropic.Anthropic(api_key=api_key)

        return client

    def run_one_prompt(self, client, row):
        user_message = self.create_user_message(row.context, row.question, row.answer_info)

        message = client.messages.create(
            model=self.model_id,
            max_tokens=1000,
            temperature=self.temperature,
            system = self.system_message,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        return message.content[0].text
