# Llama3-8×8b-MoE

[**🇨🇳中文**](./README_ZH.md) | [**🌐English**](./README.md) 

<p align="center">
    <br>
    <img src="./figures/llama3-MoE.jpg" width="800"/>
    <br>A
</p>

🖥️🖥️🖥️This project is based on the [llama3-8B-Instruct model](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Chat) released by Meta. It duplicates the MLP eight times, creates a randomly initialized router, and keeps the other parameter weights unchanged, constructing a hot-start MoE model. This approach greatly reduces the cost of training an MoE model from scratch, making it easy to quickly fine-tune and use in downstream tasks.

#### Main Contents of This Project

- 🚀 Open-source llama3-8×8b-MoE-Base/Instruct base models, which expand upon the [llama3-8B-Base/Instruct models](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) to incorporate MoE architecture.
- 🚀 Provide two versions of weights: one with a randomly initialized router and the other initialized with router weights from [chinese-mixtral ymcui/Chinese-Mixtral](https://github.com/ymcui/Chinese-Mixtral).
- 🚀 Open-source extension scripts and weight conversion scripts.
- 🚀 Conduct universal sft data fine-tuning on the built MoE model and compare it with existing MoE models.

----

## News
**🚀🚀🚀🚀 Continuously updating, please stay tuned**

**[2024/05/06] 🚀 Experiment 1: Training only the router parameters for the Llama3-8×8b-MoE-Instruct-router_randomboot version, details can be found in [Llama3-8x8b-MoE Fine-tuning Records](#1-llama3-8x8b-moe-fine-tuning-records)**

**[2024/05/04] 🚀 Open-source weight conversion scripts, uploaded versions ```Llama3-8×8b-MoE-Instruct/Base``` and ```Llama3-8×8b-MoE-Instruct/Base-router_warmboot```, feel free to use, and we also hope to receive feedback on the performance of this MoE model on specific tasks in this repository.**

**[2024/05/01] 🚀 Open-source llama3-8×8b-MoE model code, see modeling_file/modeling_llama_MoE.py, this code is applicable to llama-1/2/3.**

**[2024/04/28] 🚀 Created repository, uploaded README document**

## Content Guide
| Section                                   | Description                                                |
| ---------------------------------------- | ---------------------------------------------------------- |
| [🏰 Llama3-8x8b-MoE Fine-tuning Records](#1-llama3-8x8b-moe-fine-tuning-records) | Llama3-8x8b MoE sft Records |
| [🏰 Extending MoE Architecture and Training Experience](#2-extending-moe-architecture-experience) | Author's previous experience in building Yi-8x6b/4x4b MoE and experience with this project's llama3-8x8b MoE |
| [⏬ Model Downloads](#3-model-effects)        | Llama3-8×8b-MoE large model download links    |
| [💯 Model Effects](#4-training-and-finetuning) | Introduction to the model's performance on some tasks    |
| [📝 Training and Finetuning](#5) | Introduction to finetuning the Llama3-8×8b-MoE large model |
| [❓ Frequently Asked Questions](#) | Answers to some common questions |

## 🦳1. Llama3-8x8b-MoE Fine-tuning Records
1. Training only the router parameters (05.06, this method has not been attempted before) <br>
   
     - The parameters of the Instruct version of the MoE model have already undergone sft, and we only add the parameters of the router part. Fine-tuning all parameters has several disadvantages:<br>
        1. Fine-tuning all parameters is extremely costly.
        2. Sft is prone to catastrophic forgetting, and the model's general knowledge capacity will degrade.
        3. Models that have undergone sft are prone to overfitting when sft is performed again.<br>
   
     - Our base MoE model already has the ability to understand instructions, but due to the allocation strategy of the router, the output cannot align with human instructions. Therefore, what we need to do is train the router so that its allocation strategy can make the model follow human instructions.
     - Therefore, using a certain amount of universal sft data, freezing other parameters, and only releasing the parameters of the router, we fine-tune the Llama3-8x8b-MoE-Instruct-router_randomboot model to determine whether the model's output can follow human instructions. <br>
        1. This experiment is just a validation experiment to see if this method is feasible. (Existing MoE models have not undergone this type of experiment)
        2. The experimental data selection is [cognitivecomputations/dolphin](https://huggingface.co/datasets/cognitivecomputations/dolphin), which is a universal multitask sft data in English, and it also includes a certain amount of cot data, long text data, etc., which is more stable and less prone to overfitting compared to simple question-answering tasks.<br>
    - Experimental Results <br>
  
      🛑🛑🛑🛑 In progress, please wait
        
## 2. Extending MoE Architecture Experience
**🖥️1.1 Difference between hot-start MoE Base version and Instruct version**<br><br>

**🖥️1.2 Stability issue of router training after random initialization for Base version and Instruct version routers**<br><br>

**🖥️1.3 Can using pre-trained routers like mixtral-8x7b-MoE improve training efficiency and performance** <br>

- Based on my previous experience in building the Yi-8x6b-MoE-Instruct model, directly fine-tuning the model built with specific task data, using the Adgen (115k+1097) data as an example, at 0.1 and 0.2 epochs, router_warmboot compared to router_randomboot has obvious stability, mainly reflected in the following aspects: 
  
  1. At 0.1 epoch, router_warmboot can output complete sentences, while router_randomboot outputs a large amount of empty (padding) symbols.
  2. At 0.1 epoch, the BLEU4 values of router_warmboot and

 router_randomboot are 6.96 and 0, respectively.
  3. At 0.2 epoch, router_randomboot still has 197/1097 responses that are empty, while the rest of the responses have complete sentences. The BLEU4 values of router_warmboot and router_randomboot are 8.77 and 7.15, respectively. <br>
  
  It can be seen that router_warmboot has better training stability with a small amount of data. It is easy to understand that router_randomboot, due to random initialization, and the initial eight experts are the same, so the likelihood of selecting eight experts is almost the same, without preference. This makes the router randomly assigning tokens destroy the coherence of sentences and the logits obtained lose the semantic relationship between sentences, resulting in extremely poor output quality.  
                          
 




**🖥️1.4 Issue with token allocation strategy of Base and Instruct MoE model routers**<br><br>

## 3. Model Effects
**Explanation of Versions**
> 1. Router random initialization: hugging face Instruct/Base suffix

> 2. Router initialized with router weights from chinese-mixtral-base/Instruct: router_warmboot suffix

### Download Links

| Model Name                  |   Type   |                    Specifications                    |                    Full Size (GB)                    |
| :------------------------ | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | 
| Llama3-8x8b-MoE-Base | Base model | 8x8B | [[🤗HF https://huggingface.co/gao-NLP/Llama3-8x8b-MoE-Base]](https://huggingface.co/gao-NLP/Llama3-8x8b-MoE-Base) |
| Llama3-8x8b-MoE-Instruct | Instruct model | 8x8B |[[🤗HF https://huggingface.co/gao-NLP/Llama3-8x8b-MoE-Instruct]](https://huggingface.co/gao-NLP/Llama3-8x8b-MoE-Instruct) | 

### Model Selection Guide

Below is a comparison of models in this project and suggested usage scenarios. **For chat interactions, please choose the Instruct version.**

| Comparison           |  Llama3-8×8b-MoE-Instruct                                     | Llama3-8B-Instruct                                 |  Mixtral-8×7B-MoE-Instruct | Deepseek-MoE-Chat| Qwen1.5-MoE-chat |
| :-------------------- | :----------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------:  |  :----------------------------------------------------------: | :----------------------------------------------------------:  |
| Model Type | **Base Model** | **Instruct/Chat Model (like ChatGPT)** |
| Model Size |    8×8B                          |            8B |
| Training Type     | Causal-LM (CLM)           | Instruct fine-tuning                                                     |
| Training Method | All parameters                         | All parameters |
| Trained On | meta/Llama3-8B-Instruct |  |
| Training Corpus | |   |
| Vocabulary Size | Original vocab, 127999 | Original vocab, 127999 |


## 5. Model Effects

To evaluate the effects of relevant models, this project conducts both generation effect evaluations and objective effect evaluations (NLU type) to assess the large model from different perspectives. It is recommended that users test on tasks they are interested in and choose models adapted to specific tasks.

### Objective Effect Evaluation

#### C-Eval

[C-Eval](https://cevalbenchmark.com) is a comprehensive Chinese basic model evaluation suite, with validation and test sets containing 1.3K and 12.3K multiple-choice questions covering 52 subjects.

| Models             | Type | Valid (0-shot) | Valid (5-shot) | Test (0-shot) | Test (5-shot) |
| ------------------------ | :------------: | :------------: | :-----------: | :-----------: | :-----------: |
| **Llama3-8×8b-MoE-Instruct** |  | |  |  |  |
| **Llama3-8B-Instruct**  |  |  |  | |  |
| **Mixtral-8×7B-MoE-Instruct**  |  |  |  | |  |
| **Deepseek-MoE-Chat**  |  |  |  | |  |
| **Qwen1.5-MoE-chat**  |  |  |  | |  |



#### CMMLU



#### MMLU



#### LongBench


### Quantitative Effect Evaluation



## 6. Training and Finetuning

### Pre-training


### Instruct Fine-tuning





## 5. Frequently Asked Questions
**1. "triu_tril_cuda_template" not implemented for 'BFloat16'**

  This is a torch version issue. It has been fixed in torch version 2.1.0 and later. For torch versions before 2.1.0, there are currently three solutions:
* Method 1: In modeling_llama.py line 1095, change ```causal_mask = torch.triu(causal_mask, diagonal=1)``` to:
  ```
  causal_mask = causal_mask.to(torch.float32)#
  causal_mask = torch.triu(causal_mask, diagonal=1)
  causal_mask = causal_mask.to('cuda', dtype=torch.bfloat16)#
  ```
* Method 2: Add the following line before line 1094 in modeling_llama.py:
  ```self.register_buffer("triu0",torch.ones(sequence_length, target_length).to("cuda").triu())```  
  Then modify line 1095 from ```causal_mask = torch.triu(causal_mask, diagonal=1)``` to ```causal_mask=causal_mask*self.triu0```
* Method 3: Add the following code before loading the model:
  ```torch.set_default_tensor_type(torch.cuda.HalfTensor)```
  However, this method may cause a pin_memory error in the cuda kernel, and its feasibility depends on the specific environment.
