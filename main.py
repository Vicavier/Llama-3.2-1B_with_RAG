import os
import json
from tqdm import tqdm
from huggingface_hub import snapshot_download
from modeling_llama import LlamaForCausalLM
from tokenization_llama_fast import LlamaTokenizerFast
from rag import KnowledgeBase
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_model_downloaded(model_path):
    """
    确保模型已经下载到本地，如果没有则进行下载
    返回模型的本地路径.
    """
    if os.path.exists(model_path):
        return model_path

    try:
        print(f"Downloading model unsloth/Llama-3.2-1B-Instruct to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        snapshot_download(
            repo_id="unsloth/Llama-3.2-1B-Instruct",
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
        return model_path

    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

def load_data(file_path):
    """
    从 JSONL 文件中加载数据。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def get_next_token_probabilities(model, tokenizer, prompt, device, max_new_tokens, knowledge_base):
    """
    生成回答
    """
    pre = knowledge_base.search(prompt)
    # print("match text:" + pre)
    prompt = "Given the following text: " + pre + ". Answer question: " + prompt
    inputs = tokenizer(prompt, 
                       return_tensors="pt", 
                       max_length = 1024, 
                       truncation = True,
                       padding = 'max_length')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            pad_token_id = tokenizer.eos_token_id,
            temperature = 0.5,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
    # 解码生成的文本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[1]
    return generated_text


def save_results(results, output_file):
    """
    Save results to JSONL file.
    """
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')



if __name__ == "__main__":
    set_seed(7102)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # 从 “questions/text.jsonl” 中读取数据
    data = load_data("questions/QA_PharmaDrugSales.jsonl")
    # 确定模型已下载并且获取到正确的路径
    try:
        model_path = ensure_model_downloaded("models/Llama-3.2-1B-Instruct")
        print(f"Using model from: {model_path}")    # "models/Llama-3.2-1B-Instruct"
    except Exception as e:
        print(f"Error ensuring model availability: {e}")

    # 初始化模型和tokenizer
    print(f"Loading model...")
    try:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        tokenizer = LlamaTokenizerFast.from_pretrained(model_path, truncation_side="left")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    knowledge_base = KnowledgeBase("./knowledge", tokenizer, model, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    results = []
    for item in tqdm(data, desc="Processing questions"):
        prompt = tokenizer.apply_chat_template(
            [{
                "role" : "user",
                "content": item['question']
            }],
            tokenize=False, 
            add_generation_prompt=True
        )
        try:
            # 通过 forward passes 收集大模型的回答
            answer = get_next_token_probabilities(model, tokenizer, prompt, device, 512, knowledge_base)
            results.append({
                'id':item['id'],
                'answer': answer
            })
        except Exception as e:
            print(f"Error with prediction: {e}")

    save_results(results, "./answers/RAG.jsonl")
    print(f"Results saved")