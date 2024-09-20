import asyncio
from collections import defaultdict
from dataclasses import dataclass
import uuid
import random
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, HfArgumentParser
from tqdm.asyncio import tqdm_asyncio
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

@dataclass
class Args:
    max_samples: int = 128
    max_new_tokens: int = 1500
    temperature: float = 1.0
    constitution_dataset: str = "pbevan11/aya_redteaming_consitutional"
    repo_id: str = "multilingual-constitutional-ai"
    push_to_hub: bool = True
    tensor_parallel_size: int = 1

parser = HfArgumentParser((Args,))
args = parser.parse_args_into_dataclasses()[0]

engine_args = AsyncEngineArgs(model="CohereForAI/aya-23-8B", tensor_parallel_size=args.tensor_parallel_size)
engine = AsyncLLMEngine.from_engine_args(engine_args)

tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-23-8B")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

ds = load_dataset(args.constitution_dataset)["train"]

critiques = ds['all_critiques_translated'][0]
revisions = ds['all_revisions_translated'][0]
system_chats = {}
for language in set(ds['language']):
    with open(f'data/few_shot_translations/hf_fewshot_{language}.txt', 'r') as f:
        system_chats[language] = f.read()

STOP_SEQ = ["User:", "###", "<|endoftext|>"]
sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens, stop=STOP_SEQ)

async def generate_text(prompt):
    request_id = str(uuid.uuid4())
    async for output in engine.generate(prompt, sampling_params, request_id):
        if output.finished:
            return output.outputs[0].text

async def process_text(i, task, language):
    system_chat = system_chats.get(language)
    chat = [{"role": "system", "content": system_chat}]
    random_choice = random.choice(range(16))
    row = {"language": language}
    for prompt, prompt_key, response_key in [
        (task, "init_prompt", "init_response"),
        (critiques[random_choice], "critic_prompt", "critic_response"),
        (revisions[random_choice], "revision_prompt", "revision_response"),
    ]:
        chat.append({"role": "user", "content": prompt})
        formatted_prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in chat]) + "\nAssistant:"
        completion = await generate_text(formatted_prompt)
        
        for stop_seq in STOP_SEQ:
            if completion.endswith(stop_seq):
                completion = completion[: -len(stop_seq)].rstrip()
        
        chat.append({"role": "assistant", "content": completion})
        row[prompt_key] = prompt
        row[response_key] = completion
    return i, len(tokenizer.encode(completion)), row

def process_results(results):
    all_data = defaultdict(list)

    for result in results:
        if result:
            row_data = result[2]  # Extract the row data dictionary
            
            # Add each key-value pair in row_data to all_data
            for key, value in row_data.items():
                all_data[key].append(value)

    # Convert all_data into a dataset
    processed_dataset = Dataset.from_dict(all_data)
        
    return processed_dataset

async def main():
    batch_size = 10  # Adjust this based on your system's capabilities
    all_results = []

    print(f"Dataset type: {type(ds)}")
    print(f"First item type: {type(ds[0])}")
    print(f"First item: {ds[0]}")

    for i in range(0, len(ds), batch_size):
        batch = ds[i:i+batch_size]
        tasks = []
        for idx, row in enumerate(batch):
            try:
                if isinstance(row, dict):
                    prompt = row.get("prompt")
                    language = row.get("language")
                elif isinstance(row, str):
                    # If row is a string, we need to decide how to handle it
                    # For now, let's assume the whole string is the prompt and use a default language
                    prompt = row
                    language = "unknown"
                else:
                    print(f"Unexpected row type: {type(row)}")
                    continue

                if prompt and language:
                    tasks.append(process_text(idx + i, prompt, language))
                else:
                    print(f"Missing prompt or language for row {idx + i}")
            except Exception as e:
                print(f"Error processing row {idx + i}: {str(e)}")

        try:
            results = await tqdm_asyncio.gather(*tasks)
            all_results.extend([r for r in results if r])
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {str(e)}")

    processed_dataset = process_results(all_results)
    
    print("Processed dataset:")
    print(processed_dataset)

    if processed_dataset:
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
                "language": example["language"],  # Add language column
            }

        post_ds = processed_dataset.map(process, remove_columns=processed_dataset.column_names)

        # Shuffle the dataset
        shuffled_ds = post_ds.shuffle(seed=42)

        if args.push_to_hub:
            # Calculate the midpoint
            midpoint = len(shuffled_ds) // 2

            # Split and push to hub
            shuffled_ds.select(range(midpoint)).push_to_hub(args.repo_id, split="train_sft")
            shuffled_ds.select(range(midpoint, len(shuffled_ds))).push_to_hub(args.repo_id, split="train_prefs")

if __name__ == "__main__":
    asyncio.run(main())
