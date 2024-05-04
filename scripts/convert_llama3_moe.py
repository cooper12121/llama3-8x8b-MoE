

import os
# import fire
import json
import torch
import shutil
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file

from transformers import MixtralForCausalLM,MixtralConfig

def duplicate_mlp(
    ckpt_dir="..llama3权重",
    moe_dir="构建的moe存放路径",
    num_experts=8,
    num_experts_per_token=2,
    output_router_logits=True,
    router_aux_loss_coef=0.02,
):
    """ 
    将MLP拷贝8份实现MoE模型
    保存为safetensor格式加载时会报错：safetensors_rust.SafetensorError: Error while deserializing header: HeaderTooLarge，因此保存为torch.bin格式
    """
    os.makedirs(moe_dir, exist_ok=True)

    for filename in tqdm(os.listdir(ckpt_dir)):
        if filename in ["pytorch_model.bin.index.json", "model.safetensors.index.json"]:
            index_map = json.load(open(os.path.join(ckpt_dir, filename)))
            new_index_map = {
                "metadata": index_map["metadata"],
                "weight_map": {}
            }
            for k, v in index_map["weight_map"].items():
                if "safetensors" in filename:
                    v = "pytorch_" +  v.replace("safetensors", "bin")
                if "mlp" in k:
                    for i in range(num_experts):
                        name = k.replace("mlp", f"block_sparse_moe.experts.{i}")
                        new_index_map["weight_map"][name] = v
                else:
                    new_index_map["weight_map"][k] = v

            path = os.path.join(moe_dir, "pytorch_model.bin.index.json")
        
            if not os.path.exists(path):
                json.dump(new_index_map, open(path, "w"), indent=4, ensure_ascii=False)
        elif ".bin" in filename or ".safetensors" in filename:
            if ".bin" in filename:
                weights = torch.load(os.path.join(ckpt_dir, filename))
            else:
                weights = safe_open_weight(ckpt_dir, filename)
                # weights = {}
                # with safe_open(os.path.join(ckpt_dir, filename), framework="pt") as f:
                #     for k in f.keys():
                #         weights[k] = f.get_tensor(k)
            new_weights = {}
            for k, v in weights.items():
                if "mlp" in k:
                    for i in range(num_experts):
                        name = k.replace("mlp", f"block_sparse_moe.experts.{i}")
                        new_weights[name] = v
                else:
                    new_weights[k] = v
            if ".bin" in filename:
                path = os.path.join(moe_dir, filename)
            else:
                path = os.path.join(moe_dir, "pytorch_" + filename.replace("safetensors", "bin"))
                
            if not os.path.exists(path):
                torch.save(new_weights, path)
        elif filename == "config.json":
            config = json.load(open(os.path.join(ckpt_dir, filename)))
            config["num_local_experts"] = num_experts
            config["num_experts_per_tok"] = num_experts_per_token
            config["output_router_logits"] = output_router_logits
            config["router_aux_loss_coef"] = router_aux_loss_coef
            path = os.path.join(moe_dir, filename)
            if not os.path.exists(path):
                json.dump(config, open(path, "w"), indent=4, ensure_ascii=False)
        else:
            path = os.path.join(ckpt_dir, filename)
            if os.path.isfile(path):
                shutil.copyfile(path, os.path.join(moe_dir, filename))



        

def conver_router():
    num_experts=8,
    num_experts_per_token=2,
    output_router_logits=True,
    router_aux_loss_coef=0.02,
    """ 
    使用mixtral router实现llama3-8x8b-MoE router的热气动,，二者配置一样
    注意，这里采用的是chinese-Mixtral的router权重https://github.com/ymcui/Chinese-Mixtral

    mixtral: gate(hidden_size:4096,8,no_bias)

  

    处理逻辑，
        1. 先使用duplicate_mlp()函数复制得到专家模型
        2. 更新配置文件，为index中添加router权重
        3. 由于mixtral 和 llama3的映射方式不同，需要把mixtral的所有权重文件加载进来，映射到llama3的不同层，这里可以观察llama3参数的放置方式，一定范围的层放置在一个模型文件中

     
    """
    mixtral_model_path = "chinese-mixtral-instruct"
    llama3_moe_router_warmboot = "router热启动的模型存放位置"
    
    

    #加载index映射文件
    mixtral_index = json.load(open(os.path.join(mixtral_model_path, "model.safetensors.index.json")))
    llama3_moe_router_warmboot_index = json.load(open(os.path.join(llama3_moe_router_warmboot, "pytorch_model.bin.index.json")))


   
    


    #写index.json文件
    for k, v in mixtral_index["weight_map"].items():
        if "gate" in k:
                layer_id = get_layer_id(k)
                v_replace = transfer_value(v,layer_id)
                llama3_moe_router_warmboot_index["weight_map"][k] = v_replace

    path = os.path.join(llama3_moe_router_warmboot, "pytorch_model.bin.index.json")
    # if not os.path.exists(path):
    json.dump(llama3_moe_router_warmboot_index, open(path, "w"), indent=4, ensure_ascii=False)



    #读入3个weight文件
    """ 
    这里将llama3-MoE的模型文件全部读入后遍历Mixtral的权重更新，由于model-0004中没有layer，可以不加载
    """
 
    weights_01 = torch.load(os.path.join(llama3_moe_router_warmboot, "pytorch_model-00001-of-00004.bin"))
    weights_02 = torch.load(os.path.join(llama3_moe_router_warmboot, "pytorch_model-00002-of-00004.bin"))
    weights_03 = torch.load(os.path.join(llama3_moe_router_warmboot, "pytorch_model-00003-of-00004.bin"))




    #遍历权重文件
    for filename in tqdm(os.listdir(mixtral_model_path)):
        if ".json" in filename:continue
        if ".bin" in filename or ".safetensors" in filename:
            if ".bin" in filename:
                weights = torch.load(os.path.join(mixtral_model_path, filename))
            else:
                weights = safe_open_weight(mixtral_model_path, filename)
            
            for k, v in weights.items():
                if "gate" in k:
            
                    layer_id = get_layer_id(k)

                    #根据layer_id写入对应的模型文件
                    if layer_id<=8:
                        weights_01[k]=v
                    elif layer_id<=20:
                        weights_02[k]=v
                    else:
                        weights_03[k]=v
    # 权重文件写回
    torch.save(weights_01,os.path.join(llama3_moe_router_warmboot, "pytorch_model-00001-of-00004.bin"))
    torch.save(weights_02,os.path.join(llama3_moe_router_warmboot, "pytorch_model-00002-of-00004.bin"))
    torch.save(weights_03,os.path.join(llama3_moe_router_warmboot, "pytorch_model-00003-of-00004.bin"))                        



def get_layer_id(key:str):
    #根据k值判断所在layer_id "model.layers.9.
    return int(key.split(".")[2])

def transfer_value(v:str,layer_id):
    # 根据层号确定放置的模型文件, 同一层分在两个模型文件的额可以忽略
    # 将value的数字映射转换到对应的区间 "model-00001-of-00004.safetensors" "model-00002-of-00004.safetensors....."
    if layer_id<=8:
        return "pytorch_model-00001-of-00004.bin"
    elif layer_id<=20:
        return "pytorch_model-00002-of-00004.bin"
    elif layer_id<=31:
        return "pytorch_model-00003-of-00004.bin"
    else:
        return "pytorch_model-00004-of-00004.bin"

def safe_open_weight(model_path, filename):
    """ hf框架 safe tensor的权重加载 """   
    weights = {}
    with safe_open(os.path.join(model_path, filename), framework="pt") as f:
        for k in f.keys():
            weights[k] = f.get_tensor(k)
    return weights


def test():
    import sys
    sys.path.append("path for modeling file")
    from modeling_file.llama3_moe.modeling_llama_moe import LlamaMoEForCausalLM
    from modeling_file.llama3_moe.tokenization_llama_fast import LlamaTokenizerFast

    model_ckpt = "....."


    tokenizer = LlamaTokenizerFast.from_pretrained(model_ckpt,padding_side='left')
    # print(tokenizer)

    model = LlamaMoEForCausalLM.from_pretrained(model_ckpt,device_map="auto",use_cache=False)



    text_list = ["hello,what is your name?","你好，你叫什么名字","今天天气怎么样"]
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(text_list,return_tensors="pt", padding=True).to("cuda")
   
    output = model.generate(**inputs,pad_token_id=tokenizer.eos_token_id,max_new_tokens=100)

    print(tokenizer.batch_decode(output))


if __name__ == "__main__":

    # duplicate_mlp()
    # conver_router()
    test()