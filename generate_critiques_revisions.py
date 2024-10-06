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


def generate_text(llm, prompt):
    outputs = llm.generate([prompt], sampling_params)
    if outputs:
        return outputs[0].outputs[0].text.strip()
    return ""


def process_text(i, prompt, language, llm, tokenizer):
    # row_critiques = critiques_eng[i] if language == 'English' else critiques_translated[i]
    row_revisions = (
        revisions_eng[i] if language == "English" else revisions_translated[i]
    )

    chat = []
    row = {"language": language}

    system_chat = system_chats.get(language, [])
    system_chat = [item for sublist in system_chat for item in sublist]

    for prompt_key, response_key, next_prompt_fn in [
        ("init_prompt", "init_response", lambda: random.choice(row_revisions)),
        # ("critic_prompt", "critic_response", lambda: random.choice(row_revisions)),
        ("revision_prompt", "revision_response", lambda: None),
    ]:
        chat.append({"role": "user", "content": prompt})
        full_prompt = system_chat + chat
        formatted_prompt = tokenizer.apply_chat_template(
            full_prompt, tokenize=False, add_generation_prompt=True
        )
        completion = generate_text(llm, formatted_prompt)

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


def main():
    all_results = []
    all_data = ds[:]

    llm = LLM(
        model=args.model,
        max_model_len=8192,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    for i in tqdm(range(len(ds)), desc="Processing data"):
        try:
            prompt = all_data["prompt"][i]
            language = all_data["language"][i]
            result = process_text(i, prompt, language, llm, tokenizer)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error processing row {i}: {str(e)}")

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
                    {
                        "role": "assistant",
                        "content": example["revision_response"].strip(),
                    },
                ],
                "messages": [
                    {"role": "user", "content": example["init_prompt"].strip()},
                    {
                        "role": "assistant",
                        "content": example["revision_response"].strip(),
                    },
                ],
                "rejected": [
                    {"role": "user", "content": example["init_prompt"].strip()},
                    {"role": "assistant", "content": example["init_response"].strip()},
                ],
            }

        post_ds = processed_dataset.map(
            process, remove_columns=processed_dataset.column_names
        )
        shuffled_ds = post_ds.shuffle(seed=42)

        if args.push_to_hub:
            midpoint = len(shuffled_ds) // 2
            shuffled_ds.select(range(midpoint)).push_to_hub(
                args.repo_id, split="train_sft"
            )
            shuffled_ds.select(range(midpoint, len(shuffled_ds))).push_to_hub(
                args.repo_id, split="train_prefs"
            )


if __name__ == "__main__":
    parser = HfArgumentParser((Args,))
    args = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    ds = load_dataset(args.constitution_dataset)["train"]
    # ds = ds.filter(lambda example: example['language'] == 'English').select(range(20))

    # critiques_eng = ds['all_critiques_eng']
    revisions_eng = ds["all_revisions_eng"]
    # critiques_translated = ds['all_critiques_translated']
    revisions_translated = ds["all_revisions_translated"]

    system_chats = {}
    for language in set(ds["language"]):
        with open(f"data/few_shot_translations/hf_fewshot_{language}.txt", "r") as f:
            sys_cht_full = json.load(f)
            # Removing critiques
            new_sys_cht_full = []
            for list_of_dicts in sys_cht_full:
                new_list = [
                    list_of_dicts[0],
                    list_of_dicts[1],
                    list_of_dicts[4],
                    list_of_dicts[5],
                ]
                new_sys_cht_full.append(new_list)
            ####
            system_chats[language] = new_sys_cht_full[0:3]

    STOP_SEQ = ["User:", "###", "<|endoftext|>"]
    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_new_tokens, stop=STOP_SEQ
    )

    main()
