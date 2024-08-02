import argparse
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM
from transformers.models.opt.configuration_opt import OPTConfig
from evaluate_utils import evaluate_model, evaluate_with_harness_full
from datautils import get_calib_data
from act_aware_utils import calib_input_distribution, calib_fisher_info
from sensitivity import calib_sensitivity_ppl, calib_sensitivity_stable_rank
from quantization import rtn_quant_sequential
from binary_search import binary_search_truncation_rank, fixed_truncation_rank
import numpy as np
import wandb
from os.path import join 
import json


def count_parameters(model):
    """
    Calculate the number of parameters in a model and return the count in billions.
    
    Args:
    model: The pre-trained model instance.
    
    Returns:
    float: Number of parameters in billions.
    """
    total_params = sum(p.numel() for p in model.parameters())
    total_params_in_billion = total_params / 1e9
    return total_params_in_billion

def main(args):
    # setting random seed of numpy and torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    wandb_writer = wandb.init(project="learn-to-compress-lrd2", name=args.exp_name, config=vars(args))

    # Load model
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True, cache_dir=args.cache_dir
    )
    num_params_old = count_parameters(model)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # if "llama" in model_id or "opt" in model_id:
    #     model = model.to_bettertransformer()

    # sensitivity calibration
    calib_loader = get_calib_data(args.calib_dataset, tokenizer, model_id, 256)
    if "fisher" in args.scaling_method:
        calib_fisher_info(model, calib_loader, args, args.use_cache)
    if "abs" in args.scaling_method:
        calib_input_distribution(
            model, calib_loader, args.scaling_method, args, args.use_cache
        )

    # search best truncation rank for each layer
    if args.fix_ratio: 
        print('Using fixed compression ratio')
        fixed_truncation_rank(model, args.param_ratio_target, args)
    else:
        if args.sensitivity_metric == "ppl":
            sensitivity = calib_sensitivity_ppl(model, calib_loader, args, args.use_cache)
        elif args.sensitivity_metric == "stable_rank":
            sensitivity = calib_sensitivity_stable_rank(
                model, calib_loader, args, args.use_cache
            )

            binary_search_truncation_rank(model, sensitivity, calib_loader, args)

    # quantization
    if args.weight_quant != "none":
        if args.weight_quant == "rtn_int8":
            rtn_quant_sequential(model, 8)
        elif args.weight_quant == "rtn_int6":
            rtn_quant_sequential(model, 6)

    result = evaluate_with_harness_full(model, tokenizer, model.device, debug=False, batch_size=args.eval_bs)
    print(result)
    if not os.path.exists("output"):
        os.makedirs("output")
    with open("output/result.txt", "a+") as f:
        f.write(f"{args}\n")
        f.write(f"{result}\n")

    wandb.log({**result,'step': 0})

    num_params_new = count_parameters(model)
   
    compression_stats = { "compression_stats/new_params_billion": num_params_new, "compression_stats/old_params_billion": num_params_old, "compression_stats/compression_ratio": num_params_new / num_params_old }
    print(f"\n\n--Compression Stats---\n{json.dumps(compression_stats, indent=4)}")
    wandb.log({**compression_stats, 'step': 0})
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="train_cache",
        help="Path to cache data and models",
    )

    parser.add_argument(
        "--ppl_target",
        type=float,
        default=-1,
        help="target ppl",
    )
    parser.add_argument(
        "--param_ratio_target",
        type=float,
        default=-1,
        help="target param ratio",
    )
    parser.add_argument(
        "--act_aware",
        action="store_true",
        help="use act aware svd (ASVD)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="hyper-parameter alpha for ASVD",
    )
    parser.add_argument(
        "--n_calib_samples",
        type=int,
        default=32,
        help="number of samples used for calibration",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb", "alpaca"],
        help="calibration dataset",
    )
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="abs_mean",
        choices=["abs_mean", "abs_max", "fisher", "fisher_abs_mean"],
        help="scaling method",
    )
    parser.add_argument(
        "--sensitivity_metric",
        type=str,
        default="ppl",
        choices=["ppl", "stable_rank"],
        help="search metric",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="use cached calibration results",
    )

    parser.add_argument(
        "--fix_ratio",
        action="store_true",
        default=False,
        help="If true, uses the target ratio to be fixed across all layers",
    )

    parser.add_argument(
        "--weight_quant",
        type=str,
        default="none",
        choices=["none", "rtn_int8", "rtn_int6"],
        help="weight quantization method",
    )
    parser.add_argument(
        "--eval_mmlu",
        action="store_true",
        help="evaluate mmlu",
    )
    parser.add_argument(
        "--sigma_fuse",
        type=str,
        default="UV",
        help="sigma fuse method",
        choices=["U", "V", "UV"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=233,
        help="random seed, which can significantly affect the calibration results",
    )
    parser.add_argument(
        "--compress_kv_cache",
        action="store_true",
        help="compress kv cache by asvd for k_proj and v_proj",
    )
    parser.add_argument(
        "--kv_cache_ratio_target",
        type=float,
        default=-1,
        help="kv cache ratio",
    )


    parser.add_argument(
        "--eval_bs",
        type=int,
        default=-2,
        help="batch size for evaluation harness",
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default='asvd',
        help="name of experiment",
    )

    args = parser.parse_args()

    main(args)
