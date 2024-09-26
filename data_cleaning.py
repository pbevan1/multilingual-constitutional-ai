from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, HfFolder
import re


def upload_to_hub(dataset_dict, dataset_name):
    api = HfApi()
    
    for split, dataset in dataset_dict.items():
        dataset.set_format("arrow")
        api.create_repo(repo_id=dataset_name, repo_type="dataset", exist_ok=True)
        dataset.push_to_hub(dataset_name, split=split)

    print(f"Dataset uploaded successfully to: https://huggingface.co/datasets/{dataset_name}")


def filter_responses(example):
    # Check if prompt is at least 5 characters long
    if len(example['prompt'].strip()) < 5:
        return False

    # Check if critic_response is not empty
    if example['critic_response'] == '':
        return False
    
    # Check if 'chosen' is a list with at least two elements
    if not isinstance(example['chosen'], list) or len(example['chosen']) < 2:
        return True
    
    # Check if the second element (assistant's response) has empty content
    if example['chosen'][1].get('role') == 'assistant' and example['chosen'][1].get('content', '').strip() == '':
        return False
    
    # Check for filter words in all text fields
    for field in ['prompt', 'chosen', 'rejected', 'critic_response']:
        if isinstance(example[field], str):
            if pattern.search(example[field]):
                return False
        elif isinstance(example[field], list):
            for item in example[field]:
                if isinstance(item, dict) and 'content' in item:
                    if pattern.search(item['content']):
                        return False
    
    return True


if __name__ == "__main__":

    dataset = load_dataset("pbevan11/multilingual-constitutional-ai")
    
    # Removing rows where the model mentions 'revision' or 'revised'
    filter_words = [
        # English
        r'revision', r'revised', r'relativity',
        # Hindi
        r'संशोधन', r'संशोधित',
        # Spanish
        r'revisión', r'revisado',
        # Russian
        r'ревизия', r'пересмотренный',
        # Serbian
        r'ревизија', r'ревидирано',
        # Filipino
        r'rebisyon', r'binago',
        # French
        r'révision', r'révisé',
        # Arabic
        r'مراجعة', r'منقح'
    ]
    
    
    pattern = re.compile('|'.join(filter_words), re.IGNORECASE)
    
    # Apply the filter to all splits in the dataset
    filtered_dataset = {}
    for split in dataset.keys():
        filtered_dataset[split] = dataset[split].filter(filter_responses)
        print(f"Split: {split}")
        print(f"Original size: {len(dataset[split])}")
        print(f"Filtered size: {len(filtered_dataset[split])}")
        print(f"Removed {len(dataset[split]) - len(filtered_dataset[split])} rows")
        print()

    upload_to_hub(filtered_dataset, "pbevan11/multilingual-constitutional-preference-pairs")
    
    print("Processing and uploading complete!")
