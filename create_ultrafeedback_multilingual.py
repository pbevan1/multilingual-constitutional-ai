import json
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, create_repo
import os
from tqdm import tqdm
from google.cloud import translate_v3 as translate
from google.oauth2 import service_account
import logging
from itertools import islice


def content_length_check(item):
    # Check prompt length
    if len(item['prompt']) > 2000:
        return False
    
    # Check chosen messages
    for message in item['chosen']:
        if len(message['content']) > 2000:
            return False
    
    # Check rejected messages
    for message in item['rejected']:
        if len(message['content']) > 2000:
            return False
    
    return True

def load_filtered_dataset(num_samples=10000, seed=42):
    dataset_ufb = load_dataset("HuggingFaceH4/ultrafeedback_binarized", streaming=True)
    
    # Shuffle and filter the dataset
    filtered_dataset = (
        item for item in dataset_ufb['train_prefs'].shuffle(seed=seed, buffer_size=10000)
        if content_length_check(item)
    )
    
    # Take the first num_samples items that pass the filter
    return list(islice(filtered_dataset, num_samples))


def batch_translate(texts, target_language):
    parent = f"projects/{project_id}/locations/global"
    response = client.translate_text(
        parent=parent,
        contents=texts,
        target_language_code=target_language,
        mime_type='text/plain'
    )
    return [translation.translated_text for translation in response.translations]

def translate_items_batch(items, target_language):
    texts_to_translate = []
    text_indices = []
    
    for item_index, item in enumerate(items):
        item_texts = [item['prompt']] + [msg['content'] for msg in item['chosen']] + [msg['content'] for msg in item['rejected']]
        non_empty_texts = [text for text in item_texts if text.strip()]
        
        if non_empty_texts:
            texts_to_translate.extend(non_empty_texts)
            text_indices.extend([(item_index, i) for i in range(len(non_empty_texts))])
    
    if not texts_to_translate:
        return []
    
    translated_texts = batch_translate(texts_to_translate, target_language)
    
    if len(translated_texts) != len(texts_to_translate):
        logging.error(f"Mismatch in translation count. Expected {len(texts_to_translate)}, got {len(translated_texts)}")
        return []
    
    translated_items = []
    for item_index, item in enumerate(items):
        if item_index not in [idx[0] for idx in text_indices]:
            continue  # Skip items with no content to translate
        
        item_translations = [translated_texts[i] for i, (idx, _) in enumerate(text_indices) if idx == item_index]
        
        non_empty_count = sum(1 for text in [item['prompt']] + [msg['content'] for msg in item['chosen']] + [msg['content'] for msg in item['rejected']] if text.strip())
        
        if len(item_translations) != non_empty_count:
            logging.warning(f"Mismatch in translation count for item {item_index}. Skipping.")
            continue
        
        translated_item = item.copy()
        translation_index = 0
        
        if item['prompt'].strip():
            translated_item['prompt'] = item_translations[translation_index]
            translation_index += 1
        
        translated_chosen = []
        for message in item['chosen']:
            translated_message = message.copy()
            if message['content'].strip():
                translated_message['content'] = item_translations[translation_index]
                translation_index += 1
            translated_chosen.append(translated_message)
        translated_item['chosen'] = translated_chosen
        
        translated_rejected = []
        for message in item['rejected']:
            translated_message = message.copy()
            if message['content'].strip():
                translated_message['content'] = item_translations[translation_index]
                translation_index += 1
            translated_rejected.append(translated_message)
        translated_item['rejected'] = translated_rejected
        
        translated_items.append(translated_item)
    
    return translated_items


def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def create_dataset():
    dataset_dict = DatasetDict()
    
    for lang, lang_full in zip(target_languages, target_languages_full):
        file_path = f'data/ultrafeedback_binarized_translations/{lang.lower()}_translations.jsonl'
        data = load_jsonl(file_path)
        
        # Remove the 'messages' column from each item in the data
        for item in data:
            item.pop('messages', None)
        
        dataset = Dataset.from_list(data)
        dataset_dict[lang_full] = dataset
    
    return dataset_dict


def upload_to_huggingface(dataset_dict, repo_name):
    # Create a new repository
    api = HfApi()
    create_repo(repo_name, private=False, exist_ok=True)
    
    # Push the dataset to the Hugging Face Hub
    dataset_dict.push_to_hub(repo_name)


# Main execution
if __name__ == "__main__":
    gct_mapping = {
        'English': 'en',
        'Arabic': 'ar',
        'Filipino': 'fil',
        'French': 'fr',
        'Hindi': 'hi',
        'Russian': 'ru',
        'Serbian': 'sr',
        'Spanish': 'es'
    }
    # Load 10,000 suitable samples
    dataset_ufb = load_filtered_dataset(num_samples=10000, seed=42)
    
    print(f"Loaded {len(dataset_ufb)} suitable samples.")
    df_ufb = pd.DataFrame(dataset_ufb)

    credentials = service_account.Credentials.from_service_account_file('credentials/translate_sa_key.json')
    client = translate.TranslationServiceClient(credentials=credentials)
    
    target_languages = list(gct_mapping.values())
    target_languages_full = list(gct_mapping.keys())
    
    # Your Google Cloud project ID
    project_id = "spry-optics-420515"

    # Ensure output directory exists
    os.makedirs('data/ultrafeedback_binarized_translations', exist_ok=True)
    
    # Process the dataset for each language
    batch_size = 5  # Adjust based on your needs and API limits
    
    for lang in target_languages:
        output_file = f'data/ultrafeedback_binarized_translations/{lang.lower()}_translations.jsonl'
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            batch = []
            for item in tqdm(dataset_ufb, desc=f"Translating to {lang}"):
                batch.append(item)
                
                if len(batch) == batch_size:
                    translated_items = translate_items_batch(batch, lang)
                    for translated_item in translated_items:
                        json.dump(translated_item, outfile, ensure_ascii=False)
                        outfile.write('\n')
                    batch = []
            
            # Process any remaining items
            if batch:
                translated_items = translate_items_batch(batch, lang)
                for translated_item in translated_items:
                    json.dump(translated_item, outfile, ensure_ascii=False)
                    outfile.write('\n')
    
    print("Translation and JSONL creation complete for all languages.")
    
    repo_name = "pbevan11/ultrafeedback_binarized_multilingual"
    dataset_dict = create_dataset()
    upload_to_huggingface(dataset_dict, repo_name)
    print(f"Dataset uploaded successfully to {repo_name}")
