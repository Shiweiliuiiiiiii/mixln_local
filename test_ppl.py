import os
import time
import json
import torch
import datasets
from transformers import AutoTokenizer
# from transformers import LlamaForCausalLM
from peft_pretraining import training_utils
from loguru import logger
from peft_pretraining.modeling_llama import LlamaForCausalLM

def evaluate_ppl(max_length=1024, batch_size=16, device="cuda:0", del_index=-1):
    # Load model
    model = LlamaForCausalLM.from_pretrained("/home/lius/project/MIXLN/1b_res_post_pre_lr5e-4_4layer_of_post/model_50001", torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()
    
    model_back = LlamaForCausalLM.from_pretrained("/home/lius/project/MIXLN/1b_res_post_pre_lr5e-4_4layer_of_post/model_50001", torch_dtype=torch.float16)
    model_back = model_back.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=max_length)
    # tokenizer = AutoTokenizer.from_pretrained("/defaultShare/SA-1B/lpx_workspace/mixln-1b", model_max_length=max_length)
    tokenizer.pad_token = tokenizer.eos_token
    # Load validation dataset
    val_data = datasets.load_dataset("c4", "en", split="validation", streaming=True, trust_remote_code=True)
    # val_data = datasets.load_dataset("/defaultShare/SA-1B/hugging_face_backup/HuggingFaceFW___fineweb-edu", split='train', streaming=True)
    val_data = val_data.shuffle(seed=42)

    # Preprocessing function
    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    # Prepare dataset
    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    # Initialize variables for evaluation
    pad_idx = tokenizer.pad_token_id
    target_eval_tokens = 10_000_000
    
    results = {}
    for del_index in range(-1, 0, 1):
        temp_model = model_back
        if del_index != -1:
            temp_model.model.layers = torch.nn.ModuleList(
                [layer for i, layer in enumerate(model.model.layers) if i != del_index]
            )
        
        for layer_idx, module in enumerate(temp_model.model.layers):
            module.self_attn.layer_idx = layer_idx
        print(del_index, len(temp_model.model.layers))
        evaluated_on_tokens = 0
        total_loss = torch.tensor(0.0).to(device)
        total_batches = 1
        print('Starting evaluation loop')
        # Evaluation loop
        for batch in val_data_mapped.batch(batch_size=batch_size):
            if evaluated_on_tokens > target_eval_tokens:
                break
            total_batches += 1

            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
            with torch.no_grad():
                loss = temp_model(**batch, labels=labels).loss
            total_loss += loss.detach()

            evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item()

        total_loss = total_loss / total_batches

        # Calculate perplexity
        perplexity = torch.exp(total_loss)
        results[del_index] = perplexity.item()
        print(f"Perplexity: {perplexity}", del_index)
    with open('results_mixln_1b.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
# Example usage
if __name__ == "__main__":
    import json

    evaluate_ppl()
    # ppl = evaluate_ppl(del_index=-1)
    # print(f"Perplexity: {ppl}", -1)
