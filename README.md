# Llama3-8×8b-MoE

[**中文**](./README_ZH.md) | [**🌐English**](./README.md) 

<p align="center">
    <br>
    <img src="./figures/llama3-MoE.jpg" width="800"/>
    <br>A
</p>

This project is based on the [llama3-8B-Instruct model](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Chat) released by Meta. It duplicates the MLP 8 times as 8 experts, creates a randomly initialized router, add load balancing loss, each token will choose 2 experts during forward, and keeps the other parameter weights unchanged, constructing a warm-start MoE model. This approach greatly reduces the cost of training an MoE model from scratch, making it easy to quickly fine-tune and use in downstream tasks.

#### Main Contents of This Project

- 🚀 Open-source llama3-8×8b-MoE-Base/Instruct base models, which expand upon the [llama3-8B-Base/Instruct models](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) to incorporate MoE architecture.
- 🚀 Provide two versions of weights: one with a randomly initialized router and the other initialized with router weights from [chinese-mixtral ymcui/Chinese-Mixtral](https://github.com/ymcui/Chinese-Mixtral).
- 🚀 Open-source extension scripts and weight conversion scripts.
- 🚀 Conduct universal sft data fine-tuning on the built MoE model and compare it with existing MoE models.
- 🚀 Extending Long contexts using the YaRN method.
- 🚀 Alignment training based on RLHF, DPO, ORPO. 

----

## News
**🚀🚀🚀🚀 Continuously updating, please stay tuned** 

**[2024/06/17] 🚀 Finetune the model using a mix of [dolphin1M](https://huggingface.co/datasets/cognitivecomputations/dolphin) English sft data and [firefly1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) Chinese sft data, upload the model to [HF](https://huggingface.co/gao-NLP/Llama3-8x8b-MoE-Instruct/tree/main/Llama3-8x8b-MoE-Instruct-dolphin1M_firefly1.1M). Proceed to the next stage of experimentation: long context extension.

**[2024/05/06] 🚀 Update the experimental results for ```Experiment 1: Training only router parameters```. For details, please refer to [Llama3-8x8b-MoE Fine-tuning Records](#1-llama3-8x8b-moe-fine-tuning-records). Upload the fine-tuning script and proceed to the next phase of the experiment.**

**[2024/05/06] 🚀 ```Experiment 1: Training only the router parameters ```for the Llama3-8×8b-MoE-Instruct-router_randomboot version, details can be found in [Llama3-8x8b-MoE Fine-tuning Records](#1-llama3-8x8b-moe-fine-tuning-records)**

**[2024/05/04] 🚀 Open-source weight conversion scripts, uploaded versions ```Llama3-8×8b-MoE-Instruct/Base``` and ```Llama3-8×8b-MoE-Instruct/Base-router_warmboot```, feel free to use, and we also hope to receive feedback on the performance of this MoE model on specific tasks in this repository.**

**[2024/05/01] 🚀 Open-source llama3-8×8b-MoE model code, see modeling_file/modeling_llama_MoE.py, this code is applicable to llama-1/2/3.**

**[2024/04/28] 🚀 Created repository, uploaded README document**

## Content Guide
| Section                                   | Description                                                |
| ---------------------------------------- | ---------------------------------------------------------- |
| [🏰 Llama3-8x8b-MoE Fine-tuning Records](#1-llama3-8x8b-moe-fine-tuning-records) | Llama3-8x8b MoE sft Records |
| [🏰 Extending MoE Architecture and Training Experience](#2-extending-moe-architecture-experience) | Author's previous experience in building Yi-8x6b/4x4b MoE and experience with this project's llama3-8x8b MoE |
| [⏬ Model Downloads](#3-model-download)        | Llama3-8×8b-MoE large model download links    |
| [💯 Model Performance](#4-model-performance) | Introduction to the model's performance on some tasks    |
| [📝 Training and Finetuning](#5-training-and-finetuning) | Introduction to finetuning the Llama3-8×8b-MoE large model |
| [❓ Frequently Asked Questions](#6-frequently-asked-questions) | Answers to some common questions |

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
      1. Using the **dolphin** data, we fine-tuned only the router parameters of the 8x8b-Instruct model, and performed **C-Eval** and **MMLU** benchmark evaluations on the results of fine-tuning with **90k** and **180k** sft data, respectively. See **Llama3-8×8b-MoE-Instruct-only_TrainRouter-90k** and **Llama3-8×8b-MoE-Instruct-only_TrainRouter-180k** in [Model Performance](#4-model-Performance).
      2. Result Analysis<br>
      (1). From the evaluation results of C-Eval and MMLU, it can be seen that training only the router does not improve the capabilities of our built moe model.
      (2). In terms of response coherence, the untrained moe's responses exhibit issues such as **random answers and repetition**. After training only the router, the situation remains the same, indicating that training only the router cannot achieve instruction alignment. Below are some response examples:
      ```json 
       {
        input=[
          "An integer c is a common divisor of two integers x and y if and only if c is a divisor of x and c is a divisor of y. Which of the following sets of integers could possibly be the set of all common divisors of two integers?",
          "How do scientists use satellites to study climate change?",
          "Please write the following solution using java: Given an array of integers `nums` and an integer `target`, return _indices of the two numbers such that they add up to `target`_. You may assume that each input would have **_exactly_ one solution**, and you may not use the _same_ element twice. You can return the answer in any order. **Example 1:** **Input:** nums = \[2,7,11,15\], target = 9 **Output:** \[0,1\] **Explanation:** Because nums\[0\] + nums\[1\] == 9, we return \[0, 1\]. **Example 2:** **Input:** nums = \[3,2,4\], target = 6 **Output:** \[1,2\] **Example 3:** **Input:** nums = \[3,3\], target = 6 **Output:** \[0,1\] **Constraints:** * `2 <= nums.length <= 104` * `-109 <= nums[i] <= 109` * `-109 <= target <= 109` * **Only one valid answer exists.** **Follow-up:** Can you come up with an algorithm that is less than `O(n2)` time complexity?",
          "What is the capital of France?",
          "在上海的苹果代工厂，较低的基本工资让工人们形成了“软强制”的加班默契。加班能多拿两三千，“自愿”加班成为常态。律师提示，加班后虽能获得一时不错的报酬，但过重的工作负荷会透支身体，可能对今后劳动权利造成不利影响。 输出摘要：",
          "翻译成英文： 然而结果却丝毫未改变——荷兰队还要继续苦苦等待首个大力神杯。 答案：",
          "模仿金庸，写一段小说",
          "帮我生成商品文案 输入：意大利okbaby婴儿浴盆通用型可折叠宝宝按摩抚触台洗澡浴盆支撑架文案：",
          "下列不属于中国古代三宫殿之一的是____。\nA:岱庙天贶殿\nB:孔庙大成殿\nC:故宫太和殿\nD:承德避暑山庄",
          "用python 写一个递归算法"
        ]

        Llama3-8×8b-MoE-Instruct-router_randomeboot_output=[
          "A helpful assistant. You are a helpful assistant. You are a helpful assistant. You are a helpful assistant. You are a helpful assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.", 
          "You are a helpful assistant.assistant\n\nYou are a helpful assistant.assistant\n\nYou are a helpful assistant.assistant\n\nYou are a helpful assistant.assistant\n\nYou are a helpful assistant.assistant\n\nYou are a helpful assistant.assistant\n\nYou are a helpful assistant.assistant\n\nYou are a helpful assistant.assistant\n\nYou are a helpful assistant.",
          "Hereassistant\n\nPlease write the following solution using java: Given an array of integers `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and",
          "Iassistant\n\nIassistant\n\nIassistant\n\nIassistant\n\nIassistant\n\nIassistant\n\nIassistant\n\nIassistant\n\nIassistant\n\nIassistant\n\nIassistant\n\nIassistant\n\nIassistant\n\nIassistant\n\nIassistant\n\nIassistant\n\nIassistant\nIassistant\nIassistant\nIassistant\nI\n", 
          "Here are a helpful assistant.assistant\n\nHere are a helpful assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant", 
          "What is a helpful assistant. 一个乐于助人的助手。assistantassistantassistantassistantassistantassistantassistant",                                            "Here is a helpful assistant.assistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistant", 
          "The helpful assistant.assistant\n\nThe helpful assistant.assistant\n\nThe helpful assistant.assistant\n\nThe helpful assistant.assistant\n\nThe helpful assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.",
          "Here is a helpful assistant.assistant\n\nHere is a helpful assistant.assistant\n\n Here is a helpful assistant.assistant\n\n Here is a helpful assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant."
        ],
        Llama3-8×8b-MoE-Instruct-only_TrainRouter-90k_output=[
        "A helpful assistant. You are a helpful assistant. You are a helpful assistant. You are a helpful assistant. You are a helpful assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.",
        "Scientists use satellites to study climate change.assistant\n\nYou are a helpful assistant.assistant\n\nYou are a helpful assistant.assistant\n\nYou are a helpful assistant.assistant\n\nYou are a helpful assistant.assistant\n\nYou are a helpful assistant.assistant\n\nYou are a helpful assistant.assistant\n\nYou are a helpful assistant.assistant\n\nYou are a helpful assistant.assistant\n\nYou are a helpful assistant.",
        "Hereassistant\n\nPlease write the following solution using java: Given an array of integers `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and an integer `nums` and",
        "What is the capital of France?assistant\n\nWhat is the capital of France?assistant\n\nWhat is the capital of France?assistant\n\nWhat is the capital of France?assistant",
        "Here are a helpful assistant.assistant\n\nHere are a helpful assistant.assistant\n\nHere are a helpful assistant.assistant\n\nHere are a helpful assistant.assistant\n\n Here are a helpful assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.",
        "Here are a helpful assistant.assistant\n\nHere are a helpful assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant",
        "What is a helpful assistant. 一个乐于助人的助手。assistantassistantassistantassistantassistantassistantassistant",
        "Here is a helpful assistant.assistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistantassistant",
        "The helpful assistant.assistant\n\nThe helpful assistant.assistant\n\nThe helpful assistant.assistant\n\nThe helpful assistant.assistant\n\nThe helpful assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.assistant.",
        ]
        Llama3-8×8b-MoE-Instruct-only_TrainRouter-180k repeated similarily
        }
        ```
2. Full-parameter fine-tuning <br>
   - In order to restore the instruction-following capabilities of our built MoE, we perform sft fine-tuning using a mixed Chinese and English instruction dataset. The fine-tuning details are as follows:
   - sft data:
      [dolphin-1M](https://huggingface.co/datasets/cognitivecomputations/dolphin) English multi-task instruction dataset and [firefly-1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) Chinese multi-task instruction dataset. After shuffling and mixing the two datasets, a total of 2500k data entries are obtained.
   - Experimental results:
      The experiment is in progress, please wait 🕔🕔🕔
        
## 2. Extending MoE Architecture Experience
**1.1 Difference between hot-start MoE Base version and Instruct version**<br><br>

**1.2 Stability issue of router training after random initialization for Base version and Instruct version routers**<br><br>

**1.3 Can using pre-trained routers like mixtral-8x7b-MoE improve training efficiency and performance** <br>

- Based on my previous experience in building the Yi-8x6b-MoE-Instruct model, directly fine-tuning the model built with specific task data, using the Adgen (115k+1097) data as an example, at 0.1 and 0.2 epochs, router_warmboot compared to router_randomboot has obvious stability, mainly reflected in the following aspects: 
  
  1. At 0.1 epoch, router_warmboot can output complete sentences, while router_randomboot outputs a large amount of empty (padding) symbols.
  2. At 0.1 epoch, the BLEU4 values of router_warmboot and router_randomboot are 6.96 and 0, respectively.
  3. At 0.2 epoch, router_randomboot still has 197/1097 responses that are empty, while the rest of the responses have complete sentences. The BLEU4 values of router_warmboot and router_randomboot are 8.77 and 7.15, respectively. <br>
  
  It can be seen that router_warmboot has better training stability with a small amount of data. It is easy to understand that router_randomboot, due to random initialization, and the initial eight experts are the same, so the likelihood of selecting eight experts is almost the same, without preference. This makes the router randomly assigning tokens destroy the coherence of sentences and the logits obtained lose the semantic relationship between sentences, resulting in extremely poor output quality.  
                          
 




**1.4 Issue with token allocation strategy of Base and Instruct MoE model routers**<br><br>

## 3. Model Download
**Explanation of Versions**
> 1. Router random initialization: hugging face Instruct/Base router_randomboot suffix

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


## 4. Model Performance

To evaluate the effects of relevant models, this project conducts both generation effect evaluations and objective effect evaluations (NLU type) to assess the large model from different perspectives. It is recommended that users test on tasks they are interested in and choose models adapted to specific tasks.

### Objective Effect Evaluation

#### C-Eval

[C-Eval](https://cevalbenchmark.com) is a comprehensive Chinese basic model evaluation suite, with validation and test sets containing 1.3K and 12.3K multiple-choice questions covering 52 subjects.

| Models                   | type            | Valid (0-shot) | Valid (5-shot)|
| :------------------------: | :------------: | :------------:  |:------------|
| **Llama3-8B-Instruct**  | baseline model |  | 
| **chinese-Mixtral-8×7B-MoE-Instruct**  |  | 51.7	| 55.0 |  
| **Deepseek-MoE-Chat**  |  | 40.0 | 40.6 |
| **Qwen1.5-MoE-chat**  |  |  | 
| ----------------------- | ------------| ------------  |------------|
|**Llama3-8×8b-MoE-Instruct-router_randomboot**|                           | 51.4| 51.3|
| **Llama3-8×8b-MoE-Instruct-only_TrainRouter-90k** | only train router parameters |  51.4| 51.3 |
| **Llama3-8×8b-MoE-Instruct-only_TrainRouter-180k**|                |  51.4| 51.3 |

#### MMLU

[MMLU](https://github.com/hendrycks/test)It is an English evaluation dataset for assessing natural language understanding capabilities, in which the validation set and test set contain 1.5K and 14.1K multiple-choice questions, respectively, covering 57 subjects.

| Models                   | type            | Valid (0-shot) | Valid (5-shot)|
| :------------------------: | :------------: | :------------:  |:------------|
| **Llama3-8B-Instruct**  | baseline model |  | 
| **chinese-Mixtral-8×7B-MoE-Instruct**  |  | 65.1|	69.6 |  
| **Deepseek-MoE-Chat**  | | 47.2 | 45.0 |
| **Qwen1.5-MoE-chat**  |  | 62.5 | 
| ----------------------- | ------------| ------------  |------------|
|**Llama3-8×8b-MoE-Instruct-router_randomboot**|                           |62.2| 63.6|
| **Llama3-8×8b-MoE-Instruct-only_TrainRouter-90k** | only train router parameters |  62.2| 63.6 |
| **Llama3-8×8b-MoE-Instruct-only_TrainRouter-180k**|                |  62.2| 63.6 |


## 5. Training and Finetuning

### Pre-training


### Instruct Fine-tuning





## 6. Frequently Asked Questions
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
