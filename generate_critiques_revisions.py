import asyncio
from collections import defaultdict
from dataclasses import dataclass
import json
import random
import pandas as pd
from text_generation import AsyncClient
from transformers import AutoTokenizer, HfArgumentParser
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset, Dataset
import time
from huggingface_hub import HfApi
import logger

api = HfApi()

@dataclass
class Args:
    max_samples: int = 128
    """The maximum number of samples to generate (use -1 for all)"""
    max_new_tokens: int = 1500
    """Max new tokens"""
    temperature: float = 1.0
    """Generation temperature"""
    constitution_dataset: str = "pbevan11/aya_redteaming_consitutional"
    """HuggingFace dataset containing the constitution"""
    repo_id: str = "multilingual-constitutional-ai"
    """The repo id to push to"""
    timestamp: bool = False
    """Whether to add a timestamp to the repo_id"""
    push_to_hub: bool = True
    """Whether to push to hub"""
    tgi_url: str = "http://localhost:8080"
    """URL for the TGI server"""

parser = HfArgumentParser(Args)
args = parser.parse_args_into_dataclasses()[0]
if args.timestamp:
    args.repo_id += str(int(time.time()))

tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-23-8B")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

ds = load_dataset(args.constitution_dataset)
critiques = ds['train']['all_critiques_translated'][0]
revisions = ds['train']['all_revisions_translated'][0]
system_chats = {}
for language in set(ds['train']['language']):
    with open(f'data/few_shot_translations/hf_fewshot_{language}.txt', 'r') as f:
        system_chat = f.read()
        # system_chat = [item for sublist in system_chat for item in sublist]
        system_chats[language] = system_chat

client = AsyncClient(args.tgi_url)
STOP_SEQ = ["User:", "###", "<|endoftext|>"]

async def process_text(split, i, task, language):
    system_chat = system_chats.get(language, '')
    if isinstance(system_chat, str):
        # If system_chat is a string, we'll treat it as a single system message
        chat = [{"role": "system", "content": system_chat}]
    else:
        # If it's already a list, we'll use it as is
        chat = system_chat.copy()
    
    random_choice = random.choice(range(16))
    token_length = 0
    row = {}
    
    for prompt, prompt_key, response_key in [
        (task, "init_prompt", "init_response"),
        (critiques[random_choice], "critic_prompt", "critic_response"),
        (revisions[random_choice], "revision_prompt", "revision_response"),
    ]:
        prompt_dict = {"role": "user", "content": prompt}
        chat.append(prompt_dict)
        
        # # For debugging, print the chat structure instead of its content
        # print(f"Chat structure: {[m['role'] for m in chat]}")
        
        formatted_prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in chat])
        formatted_prompt += "\nAssistant:"
        
        try:
            completion = await client.generate(
                formatted_prompt,
                max_new_tokens=args.max_new_tokens,
                stop_sequences=STOP_SEQ,
                temperature=args.temperature,
            )
            generated_text = completion.generated_text
            for stop_seq in STOP_SEQ:
                if generated_text.endswith(stop_seq):
                    generated_text = generated_text[:-len(stop_seq)].rstrip()
            response_dict = {"role": "assistant", "content": generated_text}
            chat.append(response_dict)
            token_length += len(tokenizer.encode(generated_text))
            row[prompt_key] = prompt
            row[response_key] = generated_text
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            row[prompt_key] = prompt
            row[response_key] = f"Error: {str(e)}"
    
    return split, i, token_length, row

async def main():
    start_time = time.time()
    tasks = [process_text(split, idx, row["prompt"], row["language"]) for split in ds for idx, row in enumerate(ds[split])]
    results = await tqdm_asyncio.gather(*tasks)
    end_time = time.time()

    total_duration = end_time - start_time
    total_tokens = sum(result[2] for result in results)
    overall_tokens_per_second = total_tokens / total_duration if total_duration > 0 else 0
    print(f"Overall Tokens per Second: {overall_tokens_per_second}")
    all_ds = defaultdict(lambda: defaultdict(list))
    for result in results:
        [all_ds[result[0]][key].append(value) for key, value in result[3].items()]

    def process(example):
        return {
            "prompt": example["init_prompt"].strip(),
            "messages": [
                {"role": "user", "content": example["init_prompt"].strip()},
                {"role": "assistant", "content": example["revision_response"].strip()},
            ],
            "chosen": [
                {"role": "user", "content": example["init_prompt"].strip()},
                {"role": "assistant", "content": example["revision_response"].strip()},
            ],
            "rejected": [
                {"role": "user", "content": example["init_prompt"].strip()},
                {"role": "assistant", "content": example["init_response"].strip()},
            ],
        }

    for split in all_ds:
        df = pd.DataFrame(all_ds[split])
        print("=" * 10 + split + "=" * 10)
        print(df)
        post_ds = Dataset.from_dict(all_ds[split])
        post_ds = post_ds.map(process)
        if args.push_to_hub:
            post_ds.select(range(len(post_ds) // 2)).push_to_hub(args.repo_id, split=f"{split}_sft")
            post_ds.select(range(len(post_ds) // 2, len(post_ds))).push_to_hub(args.repo_id, split=f"{split}_prefs")
            if "/" not in args.repo_id:
                repo_id = f"{api.whoami()['name']}/{args.repo_id}"
            for file, name in zip([__file__], ["create_dataset.py"]):
                api.upload_file(
                    path_or_fileobj=file,
                    path_in_repo=name,
                    repo_id=repo_id,
                    repo_type="dataset",
                )

if __name__ == "__main__":
    asyncio.run(main())