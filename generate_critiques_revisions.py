import random
from collections import defaultdict
from dataclasses import dataclass
import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, HfArgumentParser
from vllm import LLM, SamplingParams
from tqdm import tqdm

@dataclass
class Args:
    model: str = "natong19/Mistral-Nemo-Instruct-2407-abliterated"
    max_new_tokens: int = 1024
    temperature: float = 0.3
    constitution_dataset: str = "pbevan11/aya_redteaming_consitutional"
    repo_id: str = "multilingual-constitutional-preference-pairs-2"
    push_to_hub: bool = True
    tensor_parallel_size: int = 1

def generate_texts(llm, prompts):
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text.strip() if output.outputs else "" for output in outputs]

def process_texts(all_data, llm, tokenizer):
    all_results = []
    batch_size = 1000

    for i in tqdm(range(0, len(all_data['prompt']), batch_size), desc="Processing batches"):
        batch_slice = slice(i, i + batch_size)
        batch_prompts = all_data['prompt'][batch_slice]
        batch_languages = all_data['language'][batch_slice]

        init_prompts = []
        revision_prompts = []

        for j, (prompt, language) in enumerate(zip(batch_prompts, batch_languages)):
            row_revisions = revisions_eng[i+j] if language == "English" else revisions_translated[i+j]
            
            system_chat = system_chats.get(language, [])
            system_chat = [item for sublist in system_chat for item in sublist]

            # Prepare initial prompt
            init_chat = system_chat + [{"role": "user", "content": prompt}]
            init_prompts.append(tokenizer.apply_chat_template(init_chat, tokenize=False, add_generation_prompt=True))

            # Prepare revision prompt
            revision_prompt = random.choice(row_revisions)
            revision_chat = system_chat + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""},  # Placeholder for init_response
                {"role": "user", "content": revision_prompt}
            ]
            revision_prompts.append((revision_prompt, revision_chat))

        # Generate initial responses
        init_responses = generate_texts(llm, init_prompts)

        # Update revision prompts with initial responses and generate revision responses
        final_revision_prompts = []
        for init_response, (revision_prompt, revision_chat) in zip(init_responses, revision_prompts):
            revision_chat[1]["content"] = init_response  # Update placeholder
            final_revision_prompts.append(tokenizer.apply_chat_template(revision_chat, tokenize=False, add_generation_prompt=True))

        revision_responses = generate_texts(llm, final_revision_prompts)

        # Process results
        for j, (prompt, language, init_response, revision_prompt, revision_response) in enumerate(zip(
            batch_prompts, batch_languages, init_responses, 
            [rp for rp, _ in revision_prompts], revision_responses
        )):
            row = {
                "language": language,
                "init_prompt": prompt.strip(),
                "init_response": init_response.strip(),
                "revision_prompt": revision_prompt.strip(),
                "revision_response": revision_response.strip(),
            }
            all_results.append((i+j, len(tokenizer.encode(revision_response)), row))

    return all_results

def process_results(results):
    all_data = defaultdict(list)
    for result in results:
        if result:
            row_data = result[2]
            for key, value in row_data.items():
                all_data[key].append(value)
    return Dataset.from_dict(all_data)

def main():
    all_data = ds[:]
    
    llm = LLM(
        model=args.model,
        max_model_len=8192,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    all_results = process_texts(all_data, llm, tokenizer)
    processed_dataset = process_results(all_results)

    if processed_dataset:
        def process(example):
            return {
                "language": example["language"],
                "prompt": example["init_prompt"].strip(),
                "init_response": example["init_response"].strip(),
                "revision_prompt": example["revision_prompt"].strip(),
                "revision_response": example["revision_response"].strip(),
                "chosen": [
                    {"role": "user", "content": example["init_prompt"].strip()},
                    {"role": "assistant", "content": example["revision_response"].strip()},
                ],
                "messages": [
                    {"role": "user", "content": example["init_prompt"].strip()},
                    {"role": "assistant", "content": example["revision_response"].strip()},
                ],
                "rejected": [
                    {"role": "user", "content": example["init_prompt"].strip()},
                    {"role": "assistant", "content": example["init_response"].strip()},
                ],
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

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    ds = load_dataset(args.constitution_dataset)["train"]
    revisions_eng = ds["all_revisions_eng"]
    revisions_translated = ds["all_revisions_translated"]

    system_chats = {}
    for language in set(ds["language"]):
        with open(f"data/few_shot_translations/hf_fewshot_{language}.txt", "r") as f:
            sys_cht_full = json.load(f)
            new_sys_cht_full = [
                [list_of_dicts[0], list_of_dicts[1], list_of_dicts[4], list_of_dicts[5]]
                for list_of_dicts in sys_cht_full
            ]
            system_chats[language] = new_sys_cht_full[0:3]

    STOP_SEQ = ["User:", "###", "<|endoftext|>"]
    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_new_tokens, stop=STOP_SEQ
    )

    main()
