import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
from sacrebleu.metrics import BLEU
from tqdm import tqdm
import csv
from statistics import mean


def translate_batch(source_texts, fs_source, fs_target, src_lang, tgt_lang):
    prompts = [
        {
            "role": "user",
            "content": f"Translate the following {flores_200_mapping[src_lang]} text into {flores_200_mapping[tgt_lang]}, giving only the translated response: {fs_source[0]}"
        },
        {
            "role": "assistant",
            "content": fs_target[0]
        },
        {
            "role": "user",
            "content": f"Translate the following {flores_200_mapping[src_lang]} text into {flores_200_mapping[tgt_lang]}, giving only the translated response: {fs_source[1]}"
        },
        {
            "role": "assistant",
            "content": fs_target[1]
        }
    ] + [
        {
            "role": "user",
            "content": f"Translate the following {flores_200_mapping[src_lang]} text into {flores_200_mapping[tgt_lang]}, giving only the translated response: {text}"
        }
        for text in source_texts
    ]

    full_prompts = [
        tokenizer.apply_chat_template(
            [prompt], tokenize=False, add_generation_prompt=True
        )
        for prompt in prompts
    ]
    outputs = model.generate(full_prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def evaluate_language_pair(src_lang, tgt_lang):
   bleu = BLEU()
   translations = []
   references = []
   f"Translating {src_lang} to {tgt_lang}"
   source_texts = dataset['devtest'][2:10][f'sentence_{src_lang}']
   fs_source = dataset['devtest'][:2][f'sentence_{src_lang}']
   print("source texts")
   print(source_texts)
   target_texts = dataset['devtest'][2:10][f'sentence_{tgt_lang}']
   fs_target = dataset['devtest'][:2][f'sentence_{tgt_lang}']

   print("target texts")
   print(target_texts)
   
   batch_translations = translate_batch(source_texts, fs_source, fs_target, src_lang, tgt_lang)
   print(f"translations: {batch_translations}")
   print("--------")
   translations.extend(batch_translations)
   references.extend([[text] for text in target_texts])
   bleu_score = bleu.corpus_score(translations, references)

   return bleu_score.score


if __name__ == "__main__":

    MAX_TOKENS=1024
    TEMP=0.1
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model_name = "pbevan11/Mistral-Nemo-MCAI-SFT-DPO-revision-only"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LLM(model=model_name, trust_remote_code=True, tensor_parallel_size=torch.cuda.device_count(), max_model_len=MAX_TOKENS)
    
    # Set up sampling parameters
    sampling_params = SamplingParams(temperature=TEMP, max_tokens=MAX_TOKENS)
    
    # Load FLORES-200 dataset
    dataset = load_dataset("facebook/flores", "all", trust_remote_code=True)
    
    # Define language pairs for evaluation
    languages = ["arb_Arab", "tgl_Latn", "fra_Latn", "hin_Deva", "rus_Cyrl", "srp_Cyrl", "spa_Latn"]
    flores_200_mapping = {
        'arb_Arab': 'Arabic',
        'eng_Latn': 'English',
        'tgl_Latn': 'Filipino',
        'fra_Latn': 'French',
        'hin_Deva': 'Hindi',
        'rus_Cyrl': 'Russian',
        'srp_Cyrl': 'Serbian',
        'spa_Latn': 'Spanish'
    }
    eng = "eng_Latn"
    
    
    # Evaluate all language pairs and store results
    results = []
    x_to_eng_scores = []
    eng_to_x_scores = []
    
    for lang in languages:
        # X to ENG
        x_to_eng_score = evaluate_language_pair(lang, eng)
        results.append({"source": lang, "target": eng, "bleu_score": x_to_eng_score})
        x_to_eng_scores.append(x_to_eng_score)
        
        # ENG to X
        eng_to_x_score = evaluate_language_pair(eng, lang)
        results.append({"source": eng, "target": lang, "bleu_score": eng_to_x_score})
        eng_to_x_scores.append(eng_to_x_score)
    
    # Calculate average BLEU scores
    avg_x_to_eng = mean(x_to_eng_scores)
    avg_eng_to_x = mean(eng_to_x_scores)
    
    # Add average scores to results
    results.append({"source": "Average", "target": "X-ENG", "bleu_score": avg_x_to_eng})
    results.append({"source": "Average", "target": "ENG-X", "bleu_score": avg_eng_to_x})
    
    # Write results to CSV
    csv_filename = "flores200_evaluation_results.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['source', 'target', 'bleu_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"Results have been written to {csv_filename}")
    print(f"Average BLEU score for X-ENG: {avg_x_to_eng}")
    print(f"Average BLEU score for ENG-X: {avg_eng_to_x}")
