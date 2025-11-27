# Over-prompting -- FLLM-2025 main track
### The Few-shot Dilemma: Over-prompting Large Language Models

Link to the arXiv version: https://arxiv.org/abs/2509.13196

<!-- 
Please cite the paper:
```
@incollection{tang2024fsponer,
  title={FsPONER: Few-Shot Prompt Optimization for Named Entity Recognition in Domain-Specific Scenarios},
  author={Tang, Yongjian and Hasan, Rakebul and Runkler, Thomas},
  booktitle={ECAI 2024},
  pages={3757--3764},
  year={2024},
  publisher={IOS Press}
}
``` -->

#### The overprompting phenomenon
Over-prompting, a phenomenon where excessive examples in prompts lead to diminished performance in Large Language Models (LLMs), challenges the conventional wisdom about in-context few-shot learning. 
![Flowchart of our LLM prompting framework](Paper_figure_the_fewshot_delimma_overprompting_large_language_models.pdf)


#### A few-shot framework to study over-prompting behavior  
To systematically evaluate and compare the effectiveness of different few-shot selection strategies, we outline a prompting framework based on three standard few-shot selection methods: random sampling, semantic embedding, and TF-IDF vectors.
![Flowchart of our LLM prompting framework](line_graph_5models_promise_new_weighted.pdf)
<!-- <img src="https://github.com/user-attachments/assets/d064c31c-144b-4f54-bf1c-01ce523e05e0" alt="Description of the image." width="400"/>   -->


### requirements: 
pip install transformers, torch, accelerate

## folder - data 
run all cells in ***read_data.ipynb*** to load the original PROMISE dataset, perform the train-test split, identify the closest 160 few-shot examples for each sentence in the test dataset, and save them in the ***few_shot_list*** sub-folder.

## notebook - get_prompt_list.ipynb
run all cells ***get_prompt_list.ipynb*** to construct the prompt with varying number of few-shot examples selected through different methods, and save them in the ***processed_prompts*** folder. Remember to change the path_in and save_path according to your own computer.

## folder - llm_siemens_api_gpt_mistral
run ***gpt_4o_ai_attack.ipynb***, ***gpt_35_turbo_ai_attack.ipynb***, ***mistral_7b_eva.ipynb*** to get the completion from LLMs with API keys, and save the results in the ***completions*** folder. Remember to change the path_in and save_path according to your own computer.

## folder - completions
contain the generated completions from LLMs, based on the proposed few-shot prompting methods


