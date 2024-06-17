#  Llama3-8×8b-MoE 
  
[**🇨🇳中文**](./README_ZH.md) | [**🌐English**](./README.md) 

<p align="center">
    <br>
    <img src="./figures/llama3-MoE.jpg" width="800"/>
    <br>
</p>
<!-- <p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/cooper12121/Llama3-8×8b-MoE .svg?color=blue&style=flat-square">
    <img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/cooper12121/llama3-Chinese">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/cooper12121/llama3-Chinese">
    <a href="https://app.codacy.com/gh/cooper12121/llama3-Chinese/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade"><img src="https://app.codacy.com/project/badge/Grade/142d688425494644b5b156068f55370d"/></a>
</p> -->

本项目基于Meta发布的[llama3-8B-Instruct模型](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Chat)进行开发。即将MLP复制8份做为8个专家，创建随机初始化的router，添加负载均衡损失，其余参数权重保持不变，搭建一个热启动的MoE模型。这种方式能够极大地降低从头开始训练一个MoE模型的成本，便于快速的在下游任务中微调使用。



#### 本项目主要内容

- 🚀 开源llama3-8×8b-MoE-Base/Instruct基模型，该模型在[llama3-8B-Base/Instruct模型](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)的基础上扩展为MoE架构。
- 🚀 开源扩展脚本、权重转换脚本。
- 🚀 针对搭建的MoE模型进行通用sft数据微调，与现有的MoE模型进行比较
- 🚀 使用Yarn进行长上下文扩展
- 🚀 基于RLHF、DPO、ORPO的强化学习对齐训练
- 🚀 MoE模型训练经验总结

----



## 新闻
**  🚀🚀🚀持续更新中，请等待  **
**[2024/06/17] 🚀 使用[dolphin1M](https://huggingface.co/datasets/cognitivecomputations/dolphin)英文sft数据和[firefly1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)中文sft数据混合后做finetune，结果已上传至[HF](https://huggingface.co/gao-NLP/Llama3-8x8b-MoE-Instruct/tree/main/Llama3-8x8b-MoE-Instruct-dolphin1M_firefly1.1M)仓库。 开始下一阶段的实验：长上下文扩展**

**[2024/05/16] 🚀 更新```实验1:只训练router的参数```的实验结果，详情请参考[Llama3-8x8b-MoE微调记录](#1-llama3-8x8b-moe微调记录)。上传finetune脚本。进行下一阶段实验**

**[2024/05/06] 🚀 对 Llama3-8×8b-MoE-Instruct-router_randomboot版本进行```实验1:只训练router的参数```，详情请参考[Llama3-8x8b-MoE微调记录](#1-llama3-8x8b-moe微调记录)**

**[2024/05/04] 🚀 开源权重转换脚本，上传```Llama3-8×8b-MoE-Instruct/Base-router_randomeboot``` 版本和```Llama3-8×8b-MoE-Instruct/Base-router_warmboot```版本，欢迎大家使用，也希望大家能在本仓库反馈该MoE模型在特定任务上的效果** 

**[2024/05/01] 🚀 开源llama3-8×8b-MoE模型代码，见modeling_file/modeling_llama_MoE.py, 该代码对llama-1/2/3均适用。** 

**[2024/04/28] 🚀 创建仓库，上传README文档**


## 内容导引
| 章节                                  | 描述                                                         |
| ------------------------------------- | ------------------------------------------------------------|
| [🏰 Llama3-8x8b-MoE微调记录](#1-llama3-8x8b-moe微调记录) |llama3-8x8b MoE sft 记录|
| [🏰 扩展MoE架构及训练经验](#2-扩展moe架构经验) |作者之前自建Yi-8x6b/4x4b MoE和本项目llama3-8x8b MoE的经验体会|
| [⏬模型下载](#3-模型下载)        | Llama3-8×8b-MoE大模型下载地址    |
| [💯模型效果](#4-模型效果) | 介绍了模型在部分任务上的效果    |
| [📝训练与精调](#5-训练与精调) | 介绍了精调llama3-8×8b-MoE大模型 |
| [❓常见问题](#6-常见问题) | 一些常见问题的回复 |


## 🦳1. Llama3-8x8b-Instruct-MoE-router_randomboot微调记录
1. 只训练router的参数(05.06，该方式之前未尝试过) <br>
   
     - Instruct版本的MoE模型参数已经经过了sft，而我们仅新增了router部分的参数，对所有参数进行微调有几个缺点：<br>
        1. 所有参数微调开销极大。
        2. sft容易造成灾难性遗忘、模型的通用知识能力会退化。
        3. 已经经过sft的模型，再次进行sft极易过拟合。<br>
   
     - 我们的基座MoE模型已经具备理解指令的能力，但由于router的分配策略，使得输出无法对齐人类指令，因此我们需要做的是训练router，以便router的分配策略能够使得模型遵循人类指令。
     - 因此，先使用一定量的通用sft数据，冻结其他参数，只放开router的参数 ，对Llama3-8x8b-MoE-Instruct-router_randomboot的模型进行微调， 判断模型的输出能否遵循人类指令的目的。<br>
        1. 本实验仅仅是验证性实验，判断该方法是否可行。（现有的MoE模型并没有进行这类实验）
        2. 实验数据选择[cognitivecomputations/dolphin](https://huggingface.co/datasets/cognitivecomputations/dolphin), 该数据是英文通用的多任务sft数据，也包含了一定量的cot数据、长文本数据等，相较于简单的问答，模型的训练会更稳定、不易过拟合。<br>
    - 实验结果 <br>
        1. 使用**dolphin**数据仅对8x8b-Instruct的router参数进行微调，分别对**90k**和**180k** sft数据微调的结果进行了**C-Eval**和**MMLU**基准评测，见[模型效果](#4-模型效果)中**Llama3-8×8b-MoE-Instruct-only_TrainRouter-90k和Llama3-8×8b-MoE-Instruct-only_TrainRouter-180k**
        2. 结果分析<br>
          (1). 从C-Eval和MMLU的评测结果可以看出，只训练router并不能提升我们搭建的moe模型的能力。
          (2). 对于回复连贯性来说， 未经过训练的moe的回复存在**乱答、重复**等情况，只训练router后仍然如此，说明只训练router无法进行指令对齐，下面给出回复样例：
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
        Llama3-8×8b-MoE-Instruct-only_TrainRouter-180k 同重复
        }
       ```
2. 全参微调 <br>   
   - 为了使我们搭建的MoE恢复指令遵循能力，我们使用中英文混合的指令数据进行sft微调。微调细节如下
   - sft数据：
      [dolphin-1M](https://huggingface.co/datasets/cognitivecomputations/dolphin) 英文多任务指令数据集and [firefly-1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)中文多任务指令数据集。将二者打乱混合后共计2500k条数据。
   - 实验结果：
      实验进行中，请等待🕔🕔🕔
            


## 2. 扩展MoE架构经验
**1.1 热启动MoE Base版本与Instruct版本的区别**<br><br>

**1.2 router随机初始化后Base版本和Instruct版本router的训练稳定性问题**<br><br>

**1.3 使用已经训练好的router如mixtral-8x7b-MoE, 进行router的热启动能否提升训练效率和性能** <br>

- 根据本人之前自建Yi-8x6b-MoE-Instruct 模型的经验, 使用搭建的模型直接用于特定任务数据的微调，以Adgen(115k+1097)数据为例，在0.1和0.2个epoch处，router_warmboot相比于router_random 具有明显的稳定性，主要体现在以下几个方面： 
  
  1. 0.1 epoch处router_warmboot能够输出完整的句子，而router_randomboot输出大量为空(填充符号)。
  2. 0.1 epoch处router router_warmboot和router_randomboot的bleu4值分别为6.96、0。
  3. 0.2 epoch处router_randomboot仍然有 197/1097条回复为空，其余回复句子完整。router_warmboot和router_randomboot bleu4的值分别为  8.77和7.15 。<br>
  
  可见router_warmboot在少量的数据下具有更好地训练稳定性。不难理解，router_randomboot由于随机初始化，而初始的8个专家都相同，因此其选择8个专家的可能性几乎相同，没有偏好，这使得在少量数据下，router随机分配token破坏了句子的连贯性、得到的logits丧失了句子间的语义关系，最终输出质量极差。  
                          
 




**1.4 Base和Instruct MoE模型router对token的分配策略问题**<br><br>

## 3. 模型下载
**版本解释**
> 1. router随机初始化：hugginging faceInstruct/Base router_randomboot后缀

> 2. router 使用chinese-mixtral-base/Instruct的router权重初始化：router_warmboot后缀
### 下载地址

| 模型名称                  |   类型   |                    规格                    |                    完整版 GB）                    |
| :------------------------ | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | 
| Llama3-8x8b-MoE-Base | 基座模型 | 8x8B | [[🤗HF https://huggingface.co/gao-NLP/Llama3-8x8b-MoE-Base]](https://huggingface.co/gao-NLP/Llama3-8x8b-MoE-Base) |
| Llama3-8x8b-MoE-Instruct | 指令模型 | 8x8B |[[🤗HF https://huggingface.co/gao-NLP/Llama3-8x8b-MoE-Instruct]](https://huggingface.co/gao-NLP/Llama3-8x8b-MoE-Instruct) | 

### 模型选择指引

以下是本项目的模型对比以及建议使用场景。**如需聊天交互，请选择Instruct版。**

| 对比项                |  Llama3-8×8b-MoE-Instruct                                     | Llama3-8B-Instruct                                 |  Mixtral-8×7B-MoE-Instruct | Deepseek-MoE-Chat| Qwen1.5-MoE-chat |
| :-------------------- | :----------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------:  |  :----------------------------------------------------------: | :----------------------------------------------------------:  |
| 模型类型 | **基座模型** | **指令/Chat模型（类ChatGPT）** |
| 模型大小 |    8×8B                          |            8B |
| 训练类型     | Causal-LM (CLM)           | 指令精调                                                     |
| 训练方式 | 全参数                         | 全参数 |
| 基于什么模型训练 | meta/Llama3-8B-Instruct |  |
| 训练语料 | |   |
| 词表大小 | 原版词表，127999 | 原版词表， 127999 |


## 4. 模型效果

为了评测相关模型的效果，本项目分别进行了生成效果评测和客观效果评测（NLU类），从不同角度对大模型进行评估。推荐用户在自己关注的任务上进行测试，选择适配相关任务的模型。


### 客观效果评测

#### C-Eval

[C-Eval](https://cevalbenchmark.com)是一个全面的中文基础模型评估套件，其中验证集和测试集分别包含1.3K和12.3K个选择题，涵盖52个学科。

| Models                   | 类型            | Valid (0-shot) | Valid (5-shot)|
| :------------------------: | :------------: | :------------:  |:------------|
| **Llama3-8B-Instruct**  | 基准模型 |  | 
| **chinese-Mixtral-8×7B-MoE-Instruct**  |  | 51.7	| 55.0 |  
| **Deepseek-MoE-Chat**  |  | 40.0 | 40.6 |
| **Qwen1.5-MoE-chat**  |  |  | 
| ----------------------- | ------------| ------------  |------------|
|**Llama3-8×8b-MoE-Instruct-router_randomboot**|                           | 51.4| 51.3|
| **Llama3-8×8b-MoE-Instruct-only_TrainRouter-90k** | 只训练router参数 |  51.4| 51.3 |
| **Llama3-8×8b-MoE-Instruct-only_TrainRouter-180k**|                |  51.4| 51.3 |


<!-- #### CMMLU -->



#### MMLU

[MMLU](https://github.com/hendrycks/test)是一个用于评测自然语言理解能力的英文评测数据集，其中验证集和测试集分别包含1.5K和14.1K个选择题，涵盖57个学科

| Models                   | 类型            | Valid (0-shot) | Valid (5-shot)|
| :------------------------: | :------------: | :------------:  |:------------|
| **Llama3-8B-Instruct**  | 基准模型 |  | 
| **chinese-Mixtral-8×7B-MoE-Instruct**  |  | 65.1|	69.6 |  
| **Deepseek-MoE-Chat**  |  | 47.2 | 45.0 |
| **Qwen1.5-MoE-chat**  |  |  | 62.5|
| ----------------------- | ------------| ------------  |------------|
|**Llama3-8×8b-MoE-Instruct-router_randomboot**|                           |62.2| 63.6|
| **Llama3-8×8b-MoE-Instruct-only_TrainRouter-90k** | 只训练router参数 |  62.2| 63.6 |
| **Llama3-8×8b-MoE-Instruct-only_TrainRouter-180k**|                |  62.2| 63.6 |


<!-- #### LongBench -->


<!-- ### 量化效果评测 -->



## 5. 训练与精调

### 预训练


### 指令精调





## 6. 常见问题
**1. triu_tril_cuda_template" not implemented for 'BFloat16**

  这是torch版本的问题，在torch 2.1.0之后的版本已经修复
，对于torch 2.1.0之前的版本，目前有三种解决方案
* 方法1：在modeling_llama.py line 1095  
  将```causal_mask = torch.triu(causal_mask, diagonal=1)```  
  修改为：
  ```
  causal_mask = causal_mask.to(torch.float32)#
  causal_mask = torch.triu(causal_mask, diagonal=1)
  causal_mask = causal_mask.to('cuda', dtype=torch.bfloat16)#
  ```
* 方法2：在modeling_llama.py line 1094行前添加：  
  ```self.register_buffer("triu0",torch.ones(sequence_length, target_length).to("cuda").triu())```  
  将line 1095 ```causal_mask = torch.triu(causal_mask, diagonal=1)```  
  修改为：```causal_mask=causal_mask*self.triu0```
* 方法3：在加载模型前的代码中添加
  ```torch.set_default_tensor_type(torch.cuda.HalfTensor)```
  但这种方式可能引起cuda内核的pin_memory错误，可行与否与具体的环境有关





