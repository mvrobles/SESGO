from transformers import pipeline
from huggingface_hub import login
from dotenv import load_dotenv
import argparse
import torch
import os

from runner import LLMRunner

class LlamaRunner(LLMRunner):
    def __init__(self, temperature, save_every, model_id):
        super().__init__(temperature, save_every, model_id)

    def connect(self):
        load_dotenv()
        api_key = os.getenv('API_KEY')
        if api_key:
            login(token=api_key)

        print(f"--- Loaded model {self.model_id}")

        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU detected, using CPU.")

        device = 0 if torch.cuda.is_available() else "cpu"

        model_pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
        )
        return model_pipeline


    def run_one_prompt(self, model_pipeline, row):
        user_message = self.create_user_message(row.context, row.question, row.answer_info)
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]
        outputs = model_pipeline(messages, max_new_tokens=256, temperature = self.temperature, do_sample=True)
        model_answer = outputs[0]["generated_text"][-1]['content']

        return model_answer
