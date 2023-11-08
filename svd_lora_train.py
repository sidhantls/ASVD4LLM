import argparse
import os
import torch
import torch.nn as nn

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
)

from trl import SFTTrainer

# from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset

from evaluate import evaluate_model
from modules.svd_lora_linear import SVDLoRALinear

from utils import print_gpu_memory
from datautils import get_calib_data


def calib_input_distribution(model, calib_loader):
    model.eval()
    # set hook for every Linear layers

    def hook(module, input, output):
        abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)
        module.input_abs_mean += abs_mean

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.input_abs_mean = 0
            module.register_forward_hook(hook)

    # get activation distribution
    for batch in calib_loader:
        # print(batch)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        model(**batch)


def convert_linear_to_svd_lora_linear(module, args):
    full_name_dict = {module: name for name, module in module.named_modules()}
    modules = [module]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, child in submodule.named_children():
            if isinstance(child, nn.Linear):
                full_name = full_name_dict[child]
                rank_ratio = (
                    args.msa_rank_ratio
                    if "self_attn" in full_name
                    else args.mlp_rank_ratio
                )
                svd_linear = SVDLoRALinear.from_linear(
                    child,
                    n_param_ratio=rank_ratio,
                    lora_method=args.lora_method,
                    act_aware=args.act_aware,
                )
                del child.weight
                setattr(submodule, name, svd_linear)
                print(f"convert {full_name} to svd_lora_linear ratio={rank_ratio}")
            else:
                modules.append(child)


def total_model_parameters_buffers(model):
    return sum(p.numel() for p in model.parameters()), sum(
        p.numel() for p in model.buffers()
    )


def train(model, tokenizer, train_dataset, args):
    # LoRA Config
    if args.lora_method == "reconstruct":
        peft_parameters = LoraConfig(
            lora_alpha=64,
            lora_dropout=0.1,
            r=32,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        )
    else:
        peft_parameters = None
    # Training Params
    output_dir = f"./output/{args.model_id.replace('/','_')}_svd_lora_train_{args.lora_method}_{args.msa_rank_ratio}_{args.mlp_rank_ratio}"
    train_params = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        # optim="paged_adamw_32bit",
        optim="adamw_torch",
        save_steps=1000,
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.05,
        group_by_length=True,
        lr_scheduler_type="linear",
    )

    # Training
    fine_tuning = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=train_params,
        peft_config=peft_parameters,
    )

    # Training
    fine_tuning.train()
    fine_tuning.model.save_pretrained(output_dir)
    if args.lora_method == "reconstruct":
        model = PeftModel.from_pretrained(model, output_dir)
        model = model.merge_and_unload()
        model.save_pretrained(os.path.join(output_dir, "final_merged"))


def main(args):
    # Dataset
    # data_name = "mlabonne/guanaco-llama2-1k"
    # data_name = 'SirNeural/flan_v2'
    # data_name = 'databricks/databricks-dolly-15k'

    # Model and tokenizer names
    # base_model_name = "NousResearch/Llama-2-7b-chat-hf"
    model_id = args.model_id
    # "./checkpoints/opt-1.3b-lora-mlabonne-enhanced-svd"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix for fp16

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    raw_model_parameters, raw_model_buffers = total_model_parameters_buffers(model)
    print("raw model tot: {}".format(raw_model_parameters + raw_model_buffers))
    if args.act_aware:
        cablib_dataset = "wikitext2"
        calib_loader = get_calib_data(cablib_dataset, tokenizer, model_id, 256)
        calib_input_distribution(model, calib_loader)
    print_gpu_memory("before convert_linear_to_svd_lora_linear")
    convert_linear_to_svd_lora_linear(model, args)
    torch.cuda.empty_cache()
    print_gpu_memory("after convert_linear_to_svd_lora_linear")

    svd_model_parameters, svd_model_buffers = total_model_parameters_buffers(model)
    print("svd model tot: {}".format(svd_model_parameters + svd_model_buffers))
    print(
        "tot ratio:{}".format(
            (svd_model_parameters + svd_model_buffers)
            / (raw_model_parameters + raw_model_buffers)
        )
    )
    print("train ratio: {}".format(svd_model_parameters / raw_model_parameters))

    # train_dataset = get_qat_dataset(training_data, llama_tokenizer, 256)
    data_name = "tatsu-lab/alpaca"
    # data_name = 'timdettmers/openassistant-guanaco'
    training_dataset = load_dataset(data_name, split="train")
    train(model, tokenizer, training_dataset, args)

    model_merged = model
    query = "### Human: I am depressed, what should I do?"
    text_gen = pipeline(
        task="text-generation",
        model=model_merged,
        tokenizer=tokenizer,
        max_length=200,
    )
    # output = text_gen(f"<s>[INST] {query} [/INST]")
    output = text_gen(query)
    print(output[0]["generated_text"])
    # evaluate_model(base_model, llama_tokenizer, base_model_name, 'llmqat', limit=200, eval_ppl=False)
    evaluate_model(
        model_merged,
        tokenizer,
        model_id,
        "llmqat",
        limit=200,
        eval_ppl="",
        num_fewshot=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--msa_rank_ratio",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--mlp_rank_ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--lora_method",
        type=str,
        default="UV",
        help="lora method, default: UV",
    )
    parser.add_argument(
        "--act_aware",
        action="store_true",
        help="use act aware svd lora",
    )
    args = parser.parse_args()

    main(args)
