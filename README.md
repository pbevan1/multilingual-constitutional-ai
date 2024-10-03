<p align="center">
  <img src="assets/robot_image.jpg" width="210">
</p>

<p align="center">
    🤗 <a href="https://huggingface.co/collections/pbevan11/multilingual-constitutional-ai-66fec15dee0e7cf2faf10437" target="_blank">Models & Datasets</a> | 📃 <a href="https://sites.google.com/view/multilingual-constitutional-ai" target="_blank">Blog Post</a>
</p>

# Multilingual Constitutional AI

This work explores the application of Constitutional AI (CAI) techniques in a multilingual context. This work covers 8 languages (Arabic, Filipino, French, Hindi, English, Russian, Serbian, Spanish).

## How to navigate this project

* `construct_principles.ipynb` is a notebook that walks through the adaptation of Anthropic's constitution to create targeted critiques and revisions to the [Aya redteaming](https://huggingface.co/datasets/CohereForAI/aya_redteaming) dataset.
* `create_ultrafeedback_multilingual.py` is a script to translate Ultrafeedback Binarized into our 8 languages using NLLB-3.3B.
* `generate_critiques_revisions.py` is an optimised vLLM script which generates the constitutional preference pairs via critiquing and revising the LLM output.
* `data_cleaning.py` is a script that helps us to remove some of the unwanted examples that resulted from the `generate_critiques_revisions.py` script.
* `finetuning` contains scripts and configs to do supervised finetuning and DPO for both the safety trained model and the baseline.
* `evaluate.py` is a script that generates outputs on the test set of the red team prompts, and uses GPT-4o for LLM-as-a-judge to categorise each as either HARMFUL or HARMLESS. We also provide an explanation of the categorisation for interpretability.
* plots.ipynb is a notebook used for generating hte plots shown in the [blog post](https://sites.google.com/view/multilingual-constitutional-ai).
