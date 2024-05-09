#  Llama3-8×8b-MoE 
  
[**🇨🇳中文**](./README.md) | [**🌐English**](./README_EN.md) 

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

🖥️🖥️🖥️本项目基于Meta发布的[llama3-8B-Instruct模型](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Chat)进行开发。即将MLP复制8份，创建一个随机初始化的router，其余参数权重保持不变，搭建一个热启动的MoE模型。这种方式能够极大地降低从头开始训练一个MoE模型的成本，便于快速的在下游任务中微调使用。



#### 本项目主要内容

- 🚀 开源llama3-8×8b-MoE-Base/Instruct基模型，该模型在[llama3-8B-Base/Instruct模型](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)的基础上扩展为MoE架构。
- 🚀 提供router随机初始化和使用chinese-mixtral [ymcui/Chinese-Mixtral](https://github.com/ymcui/Chinese-Mixtral)的router权重进行初始化的两个版本的权重
- 🚀 开源扩展脚本、权重转换脚本。
- 🚀 针对搭建的MoE模型进行通用sft数据微调，与现有的MoE模型进行比较

----



## 新闻
**🚀🚀🚀🚀 持续更新中，请等待**

**[2024/05/06] 🚀 对 Llama3-8×8b-MoE-Instruct-router_randomboot版本进行```实验1:只训练router的参数```，详情请参考[Llama3-8x8b-MoE微调记录](#1-llama3-8x8b-moe微调记录)**

**[2024/05/04] 🚀 开源权重转换脚本，上传```Llama3-8×8b-MoE-Instruct/Base``` 版本和```Llama3-8×8b-MoE-Instruct/Base-router_warmboot```版本，欢迎大家使用，也希望大家能在本仓库反馈该MoE模型在特定任务上的效果** 

**[2024/05/01] 🚀 开源llama3-8×8b-MoE模型代码，见modeling_file/modeling_llama_MoE.py, 该代码对llama-1/2/3均适用。** 

**[2024/04/28] 🚀 创建仓库，上传README文档**


## 内容导引
| 章节                                  | 描述                                                         |
| ------------------------------------- | ------------------------------------------------------------|
| [🏰 Llama3-8x8b-MoE微调记录](#1-llama3-8x8b-moe微调记录) |llama3-8x8b MoE sft 记录|
| [🏰 扩展MoE架构及训练经验](#2-扩展moe架构经验) |作者之前自建Yi-8x6b/4x4b MoE和本项目llama3-8x8b MoE的经验体会|
| [⏬模型下载](#3-模型效果)        | Llama3-8×8b-MoE大模型下载地址    |
| [💯模型效果](#4-训练与精调) | 介绍了模型在部分任务上的效果    |
| [📝训练与精调](#5) | 介绍了精调llama3-8×8b-MoE大模型 |
| [❓常见问题](#) | 一些常见问题的回复 |


## 🦳1. Llama3-8x8b-MoE微调记录
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
  
      🛑🛑🛑🛑 正在进行，请等待
        
      




## 2. 扩展MoE架构经验
**🖥️1.1 热启动MoE Base版本与Instruct版本的区别**<br><br>

**🖥️1.2 router随机初始化后Base版本和Instruct版本router的训练稳定性问题**<br><br>

**🖥️1.3 使用已经训练好的router如mixtral-8x7b-MoE, 进行router的热启动能否提升训练效率和性能** <br>

- 根据本人之前自建Yi-8x6b-MoE-Instruct 模型的经验, 使用搭建的模型直接用于特定任务数据的微调，以Adgen(115k+1097)数据为例，在0.1和0.2个epoch处，router_warmboot相比于router_random 具有明显的稳定性，主要体现在以下几个方面： 
  
  1. 0.1 epoch处router_warmboot能够输出完整的句子，而router_randomboot输出大量为空(填充符号)。
  2. 0.1 epoch处router router_warmboot和router_randomboot的bleu4值分别为6.96、0。
  3. 0.2 epoch处router_randomboot仍然有 197/1097条回复为空，其余回复句子完整。router_warmboot和router_randomboot bleu4的值分别为  8.77和7.15 。<br>
  
  可见router_warmboot在少量的数据下具有更好地训练稳定性。不难理解，router_randomboot由于随机初始化，而初始的8个专家都相同，因此其选择8个专家的可能性几乎相同，没有偏好，这使得在少量数据下，router随机分配token破坏了句子的连贯性、得到的logits丧失了句子间的语义关系，最终输出质量极差。  
                          
 




**🖥️1.4 Base和Instruct MoE模型router对token的分配策略问题**<br><br>

## 3. 模型下载
**版本解释**
> 1. router随机初始化：hugginging faceInstruct/Base后缀

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


## 5. 模型效果

为了评测相关模型的效果，本项目分别进行了生成效果评测和客观效果评测（NLU类），从不同角度对大模型进行评估。推荐用户在自己关注的任务上进行测试，选择适配相关任务的模型。


### 客观效果评测

#### C-Eval

[C-Eval](https://cevalbenchmark.com)是一个全面的中文基础模型评估套件，其中验证集和测试集分别包含1.3K和12.3K个选择题，涵盖52个学科。

| Models             | 类型 | Valid (0-shot) | Valid (5-shot) | Test (0-shot) | Test (5-shot) |
| ------------------------ | :------------: | :------------: | :-----------: | :-----------: | :-----------: |
| **Llama3-8×8b-MoE-Instruct** |  | |  |  |  |
| **Llama3-8B-Instruct**  |  |  |  | |  |
| **Mixtral-8×7B-MoE-Instruct**  |  |  |  | |  |
| **Deepseek-MoE-Chat**  |  |  |  | |  |
| **Qwen1.5-MoE-chat**  |  |  |  | |  |



#### CMMLU



#### MMLU



#### LongBench


### 量化效果评测



## 6. 训练与精调

### 预训练


### 指令精调





## 5. 常见问题
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





