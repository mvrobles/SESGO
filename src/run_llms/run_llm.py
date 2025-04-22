import argparse
from deepseek_runner import DeepseekRunner
from llama_runner import LlamaRunner
from gpt_runner import GPTRunner
from gemini_runner import GeminiRunner
from claude_runner import ClaudeRunner

model_ids = {
    'llama': "meta-llama/Llama-3.1-8B-Instruct",
    'deepseek': "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    'llama_uncensored': "Orenguteng/Llama-3.1-8B-Lexi-Uncensored",
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LLM on Excel data')
    parser.add_argument('--model_id', type=str, required=True, help='Model ID: llama, llama_uncensored, gpt, deepseek, gemini')
    parser.add_argument('--excel_path', type=str, required=True, help='Path to the Excel file')
    parser.add_argument('--sheet_name', type=str, required=True, help='Sheet name in the Excel file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output CSV file')
    parser.add_argument('--temperature', type=float, required=True, help='Value of model temperature')
    parser.add_argument('--save_every', type=int, default=10, help='Save progress every N rows')

    args = parser.parse_args()

    if args.model_id == 'llama':
        runner = LlamaRunner(args.temperature, args.save_every, model_id = model_ids['llama'])
    elif args.model_id == 'llama_uncensored':
        runner = LlamaRunner(args.temperature, args.save_every, model_id = model_ids['llama_uncensored'])
    elif args.model_id == 'deepseek':
        runner = DeepseekRunner(args.temperature, args.save_every, model_id = model_ids['deepseek'])
    elif args.model_id == 'gpt':
        runner = GPTRunner(args.temperature, args.save_every, model_id = 'gpt')
    elif args.model_id == 'gemini':
        runner = GeminiRunner(args.temperature, args.save_every, model_id = "gemini-2.0-flash")
    elif args.model_id == 'claude':
        runner = ClaudeRunner(args.temperature, args.save_every, model_id = 'claude-3-5-haiku-20241022')
    else:
        raise ValueError("Invalid model ID. Choose from: llama, llama_uncensored, gpt, deepseek, gemini")
    
    df = runner.read_excel(args.excel_path, args.sheet_name)
    df = runner.process_excel(df)

    print('Processed prompts')

    model_pipeline = runner.connect()

    print(f'Starting to process {len(df)} prompts. \nHead:')
    print(df.head())

    runner.run_llm(model_pipeline, df, args.output_path)