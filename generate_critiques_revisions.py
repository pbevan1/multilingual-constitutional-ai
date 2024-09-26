import asyncio
from collections import defaultdict
from dataclasses import dataclass
import json
import uuid
import random
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, HfArgumentParser
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

@dataclass
class Args:
    model: str = "natong19/Mistral-Nemo-Instruct-2407-abliterated"
    max_samples: int = 128
    max_new_tokens: int = 1024
    temperature: float = 0.3
    constitution_dataset: str = "pbevan11/aya_redteaming_consitutional"
    repo_id: str = "multilingual-constitutional-ai"
    push_to_hub: bool = True
    tensor_parallel_size: int = 1

async def generate_text(prompt):
    request_id = str(uuid.uuid4())
    async for output in engine.generate(prompt, sampling_params, request_id):
        if output.finished:
            return output.outputs[0].text

async def process_text(i, prompt, language):
    row_critiques = critiques_eng[i] if language == 'English' else critiques_translated[i]
    row_revisions = revisions_eng[i] if language == 'English' else revisions_translated[i]

    chat = []
    row = {"language": language}

    system_chat = system_chats.get(language, [])
    system_chat = [item for sublist in system_chat for item in sublist]
    
    for prompt_key, response_key, next_prompt_fn in [
        ("init_prompt", "init_response", lambda: random.choice(row_critiques)),
        ("critic_prompt", "critic_response", lambda: random.choice(row_revisions)),
        ("revision_prompt", "revision_response", lambda: None),
    ]:
        chat.append({"role": "user", "content": prompt})
        full_prompt = system_chat + chat
        formatted_prompt = tokenizer.apply_chat_template(full_prompt, tokenize=False, add_generation_prompt=True)
        completion = await generate_text(formatted_prompt)
        
        for stop_seq in STOP_SEQ:
            if completion.endswith(stop_seq):
                completion = completion[: -len(stop_seq)].rstrip()
        
        chat.append({"role": "assistant", "content": completion})
        row[prompt_key] = prompt
        row[response_key] = completion
        prompt = next_prompt_fn()
    
    return i, len(tokenizer.encode(chat[-1]["content"])), row

def process_results(results):
    all_data = defaultdict(list)
    for result in results:
        if result:
            row_data = result[2]
            for key, value in row_data.items():
                all_data[key].append(value)
    return Dataset.from_dict(all_data)

async def main():
    all_results = []
    all_data = ds[:]

    for i in range(len(ds)):
        try:
            prompt = all_data['prompt'][i]
            language = all_data['language'][i]
            result = await process_text(i, prompt, language)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error processing row {i}: {str(e)}")

    processed_dataset = process_results(all_results)

    if processed_dataset:
        def process(example):
            return {
                "prompt": example["init_prompt"].strip(),
                "chosen": [
                    {"role": "user", "content": example["init_prompt"].strip()},
                    {"role": "assistant", "content": example["revision_response"].strip()},
                ],
                "rejected": [
                    {"role": "user", "content": example["init_prompt"].strip()},
                    {"role": "assistant", "content": example["init_response"].strip()},
                ],
                "language": example["language"],
                "critic_response": example["critic_response"].strip(),
            }

        post_ds = processed_dataset.map(process, remove_columns=processed_dataset.column_names)
        shuffled_ds = post_ds.shuffle(seed=42)

        if args.push_to_hub:
            midpoint = len(shuffled_ds) // 2
            shuffled_ds.select(range(midpoint)).push_to_hub(args.repo_id, split="train_sft")
            shuffled_ds.select(range(midpoint, len(shuffled_ds))).push_to_hub(args.repo_id, split="train_prefs")


if __name__ == "__main__":
    parser = HfArgumentParser((Args,))
    args = parser.parse_args_into_dataclasses()[0]
    
    engine_args = AsyncEngineArgs(model=args.model, tensor_parallel_size=args.tensor_parallel_size, max_model_len=4096)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    ds = load_dataset(args.constitution_dataset)["train"]
    
    critiques_eng = ds['all_critiques_eng']
    revisions_eng = ds['all_revisions_eng']
    critiques_translated = ds['all_critiques_translated']
    revisions_translated = ds['all_revisions_translated']

    system_chats = {}
    for language in set(ds['language']):
        with open(f'data/few_shot_translations/hf_fewshot_{language}.txt', 'r') as f:
            system_chats[language] = json.load(f)
    
    STOP_SEQ = ["User:", "###", "<|endoftext|>"]
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens, stop=STOP_SEQ)
    
    asyncio.run(main())