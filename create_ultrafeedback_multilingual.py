import json
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import create_repo
import os
import pandas as pd
from tqdm import tqdm
import torch
from itertools import islice
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM


def content_length_check(item):
    if len(item["prompt"]) > 2000:
        return False

    for message in item["chosen"]:
        if len(message["content"]) > 2000:
            return False

    for message in item["rejected"]:
        if len(message["content"]) > 2000:
            return False

    return True


def load_filtered_dataset(num_samples=10000, seed=42):
    dataset_ufb = load_dataset("HuggingFaceH4/ultrafeedback_binarized", streaming=True)

    # Shuffle and filter the dataset
    filtered_dataset = (
        item
        for item in dataset_ufb["train_prefs"].shuffle(seed=seed, buffer_size=10000)
        if content_length_check(item)
    )

    # Take the first num_samples items that pass the filter
    return list(islice(filtered_dataset, num_samples))


def translate(
    model, tokenizer, text, target_language, source_language="English", max_tokens=512
):
    translation = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=flores_200_mapping[source_language],
        tgt_lang=flores_200_mapping[target_language],
        max_length=max_tokens,
        device=0,
    )
    result = translation(text)
    return result[0]["translation_text"]


def translate_item(model, tokenizer, item, target_language):
    translated_item = item.copy()

    # Translate the prompt once
    translated_prompt = translate(
        model, tokenizer, item["prompt"], target_language, max_tokens=1024
    )
    translated_item["prompt"] = translated_prompt

    # Translate 'chosen' content
    translated_chosen = []
    for message in item["chosen"]:
        translated_message = message.copy()
        if message["content"] == item["prompt"]:
            # Reuse the translated prompt if the content matches the original prompt
            translated_message["content"] = translated_prompt
        else:
            translated_message["content"] = translate(
                model, tokenizer, message["content"], target_language, max_tokens=1024
            )
        translated_chosen.append(translated_message)
    translated_item["chosen"] = translated_chosen

    # Translate 'rejected' content
    translated_rejected = []
    for message in item["rejected"]:
        translated_message = message.copy()
        if message["content"] == item["prompt"]:
            # Reuse the translated prompt if the content matches the original prompt
            translated_message["content"] = translated_prompt
        else:
            translated_message["content"] = translate(
                message["content"], target_language, max_tokens=1024
            )
        translated_rejected.append(translated_message)
    translated_item["rejected"] = translated_rejected

    # Remove 'messages' field
    translated_item.pop("messages", None)

    return translated_item


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def create_dataset():
    dataset_dict = DatasetDict()

    for lang, lang_full in zip(target_languages, target_languages):
        file_path = f"data/ultrafeedback_binarized_translations/{lang.lower()}_translations.jsonl"
        data = load_jsonl(file_path)

        # Remove the 'messages' column from each item in the data
        for item in data:
            item.pop("messages", None)

        dataset = Dataset.from_list(data)
        dataset_dict[lang_full] = dataset

    return dataset_dict


def upload_to_huggingface(dataset_dict, repo_name):
    create_repo(repo_name, private=False, exist_ok=True)
    dataset_dict.push_to_hub(repo_name)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint = "facebook/nllb-200-3.3B"
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, clean_up_tokenization_spaces=True
    )

    flores_200_mapping = {
        "Arabic": "arb_Arab",
        "English": "eng_Latn",
        "Filipino": "tgl_Latn",
        "French": "fra_Latn",
        "Hindi": "hin_Deva",
        "Russian": "rus_Cyrl",
        "Serbian": "srp_Cyrl",
        "Spanish": "spa_Latn",
    }

    target_languages = [lang for lang in flores_200_mapping.keys() if lang != "English"]

    dataset_ufb = load_filtered_dataset(num_samples=10000, seed=42)

    print(f"Loaded {len(dataset_ufb)} suitable samples.")
    df_ufb = pd.DataFrame(dataset_ufb)

    os.makedirs("data/ultrafeedback_binarized_translations", exist_ok=True)

    # Process the dataset for each language
    for lang in target_languages:
        output_file = f"data/ultrafeedback_binarized_translations/{lang.lower()}_translations.jsonl"

        with open(output_file, "w", encoding="utf-8") as jsonl_file:
            for item in tqdm(dataset_ufb, desc=f"Translating to {lang}"):
                translated_item = translate_item(model, tokenizer, item, lang)
                json.dump(translated_item, jsonl_file, ensure_ascii=False)
                jsonl_file.write("\n")

    print("Translation and JSONL creation complete for all languages.")

    repo_name = "pbevan11/ultrafeedback_binarized_multilingual"
    dataset_dict = create_dataset()
    upload_to_huggingface(dataset_dict, repo_name)
    print(f"Dataset uploaded successfully to {repo_name}")
