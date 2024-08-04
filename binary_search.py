import os
import torch
import torch.nn as nn
from evaluate_utils import evaluate_model, evaluate_perplexity
from modules.svd_linear import SVDLinear,GradSVDLinear
from tqdm import tqdm
import json

def binary_search_truncation_rank(model, sensitivity_dict, calib_loader, args):
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_list = []
    if args.compress_kv_cache:
        # after compression, we have 2 matrices, so kv_cache reduction is the twice of weight reduction
        args.param_ratio_target=args.kv_cache_ratio_target*2
        sensitivity_dict={k:v for k,v in sensitivity_dict.items() if "k_proj" in k or "v_proj" in k}
    for layername, v in sensitivity_dict.items():
        for ratio, ppl in v.items():
            sensitivity_list.append((layername, ratio, ppl))
    sorted_sensitive_list = sorted(sensitivity_list, key=lambda x: -x[2])

    # binary search
    high = len(sorted_sensitive_list) - 1
    low = 0
    assert args.ppl_target > 0 or args.param_ratio_target > 0

    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    while low < high:
        mid = (low + high) // 2
        layers_min_ratio = {layername: 1 for layername in sensitivity_dict.keys()}
        for layername, ratio, ppl in sorted_sensitive_list[mid:]:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], ratio)
        tot_params = 0
        compress_params = 0
        if args.ppl_target > 0:
            for layername, ratio in layers_min_ratio.items():
                raw_linear = module_dict[layername]
                info = linear_info[raw_linear]
                svd_linear = SVDLinear.from_linear(
                    raw_linear,
                    param_ratio=ratio,
                    alpha=args.alpha,
                    act_aware=args.act_aware,
                    sigma_fuse=args.sigma_fuse,
                )
                setattr(info["father"], info["name"], svd_linear)
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * ratio
            ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
            param_ratio = compress_params / tot_params
            msg = f"low={low} mid={mid}, high={high}, ppl={ppl}, param_ratio={param_ratio}"
            print(msg)
            if ppl < args.ppl_target:
                high = mid
            else:
                low = mid + 1
        else:
            for layername, ratio in layers_min_ratio.items():
                raw_linear = module_dict[layername]
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * ratio
            param_ratio = compress_params / tot_params
            msg = f"low={low} mid={mid}, high={high}, param_ratio={param_ratio}({compress_params}/{tot_params})"
            print(msg)
            if param_ratio > args.param_ratio_target:
                high = mid
            else:
                low = mid + 1

    print(f"Searching finished, decomposing layers...")
    layers_min_ratio = {layername: 1 for layername in sensitivity_dict.keys()}
    for layername, ratio, ppl in sorted_sensitive_list[mid:]:
        layers_min_ratio[layername] = min(layers_min_ratio[layername], ratio)
    for layername, ratio in tqdm(layers_min_ratio.items()):
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        svd_linear = SVDLinear.from_linear(
            raw_linear,
            param_ratio=ratio,
            alpha=args.alpha,
            act_aware=args.act_aware,
            sigma_fuse=args.sigma_fuse,
        )
        setattr(info["father"], info["name"], svd_linear)



def binary_search_truncation_rank_optimize_scale(model, sensitivity_dict, calib_loader, args):
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_list = []
    for layername, v in sensitivity_dict.items():
        for ratio, ppl in v.items():
            sensitivity_list.append((layername, ratio, ppl))
    sorted_sensitive_list = sorted(sensitivity_list, key=lambda x: -x[2])

    # binary search
    high = len(sorted_sensitive_list) - 1
    low = 0
    assert args.ppl_target > 0 or args.param_ratio_target > 0

    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    while low < high:
        mid = (low + high) // 2
        layers_min_ratio = {layername: 1 for layername in sensitivity_dict.keys()}
        for layername, ratio, ppl in sorted_sensitive_list[mid:]:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], ratio)
        tot_params = 0
        compress_params = 0
        if args.ppl_target > 0:
            for layername, ratio in layers_min_ratio.items():
                raw_linear = module_dict[layername]
                info = linear_info[raw_linear]
                svd_linear = GradSVDLinear.from_linear(
                    raw_linear,
                    param_ratio=ratio,
                    alpha=args.alpha,
                    act_aware=args.act_aware,
                    sigma_fuse=args.sigma_fuse,
                )
                setattr(info["father"], info["name"], svd_linear)
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * ratio
            ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
            param_ratio = compress_params / tot_params
            msg = f"low={low} mid={mid}, high={high}, ppl={ppl}, param_ratio={param_ratio}"
            print(msg)
            if ppl < args.ppl_target:
                high = mid
            else:
                low = mid + 1
        else:
            for layername, ratio in layers_min_ratio.items():
                raw_linear = module_dict[layername]
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * ratio
            param_ratio = compress_params / tot_params
            msg = f"low={low} mid={mid}, high={high}, param_ratio={param_ratio}({compress_params}/{tot_params})"
            print(msg)
            if param_ratio > args.param_ratio_target:
                high = mid
            else:
                low = mid + 1

    print(f"Searching finished, decomposing layers...")
    layers_min_ratio = {layername: 1 for layername in sensitivity_dict.keys()}
    for layername, ratio, ppl in sorted_sensitive_list[mid:]:
        layers_min_ratio[layername] = min(layers_min_ratio[layername], ratio)
    for layername, ratio in tqdm(layers_min_ratio.items()):
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        svd_linear = GradSVDLinear.from_linear(
            raw_linear,
            param_ratio=ratio,
            alpha=args.alpha,
            act_aware=args.act_aware,
            sigma_fuse=args.sigma_fuse,
        )
        setattr(info["father"], info["name"], svd_linear)




def binary_search_truncation_rank(model, sensitivity_dict, calib_loader, args):
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_list = []
    if args.compress_kv_cache:
        # after compression, we have 2 matrices, so kv_cache reduction is the twice of weight reduction
        args.param_ratio_target=args.kv_cache_ratio_target*2
        sensitivity_dict={k:v for k,v in sensitivity_dict.items() if "k_proj" in k or "v_proj" in k}
    for layername, v in sensitivity_dict.items():
        for ratio, ppl in v.items():
            sensitivity_list.append((layername, ratio, ppl))
    sorted_sensitive_list = sorted(sensitivity_list, key=lambda x: -x[2])

    # binary search
    high = len(sorted_sensitive_list) - 1
    low = 0
    assert args.ppl_target > 0 or args.param_ratio_target > 0

    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    rates = []
    while low < high:
        mid = (low + high) // 2
        layers_min_ratio = {layername: 1 for layername in sensitivity_dict.keys()}
        for layername, ratio, ppl in sorted_sensitive_list[mid:]:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], ratio)
        tot_params = 0
        compress_params = 0
        if args.ppl_target > 0:
            for layername, ratio in layers_min_ratio.items():
                raw_linear = module_dict[layername]
                info = linear_info[raw_linear]
                svd_linear = SVDLinear.from_linear(
                    raw_linear,
                    param_ratio=ratio,
                    alpha=args.alpha,
                    act_aware=args.act_aware,
                    sigma_fuse=args.sigma_fuse,
                )
                setattr(info["father"], info["name"], svd_linear)
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * ratio
            ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
            param_ratio = compress_params / tot_params
            msg = f"low={low} mid={mid}, high={high}, ppl={ppl}, param_ratio={param_ratio}"
            print(msg)
            if ppl < args.ppl_target:
                high = mid
            else:
                low = mid + 1
        else:
            for layername, ratio in layers_min_ratio.items():
                raw_linear = module_dict[layername]
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * ratio
            param_ratio = compress_params / tot_params
            msg = f"low={low} mid={mid}, high={high}, param_ratio={param_ratio}({compress_params}/{tot_params})"
            print(msg)
            if param_ratio > args.param_ratio_target:
                high = mid
            else:
                low = mid + 1

    print(f"Searching finished, decomposing layers...")
    layers_min_ratio = {layername: 1 for layername in sensitivity_dict.keys()}
    for layername, ratio, ppl in sorted_sensitive_list[mid:]:
        layers_min_ratio[layername] = min(layers_min_ratio[layername], ratio)
        
    from collections import defaultdict; rates = defaultdict(dict) 
    for layername, ratio in tqdm(layers_min_ratio.items()):
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        svd_linear, rank = SVDLinear.from_linear(
            raw_linear,
            param_ratio=ratio,
            alpha=args.alpha,
            act_aware=args.act_aware,
            sigma_fuse=args.sigma_fuse,
        )

        setattr(info["father"], info["name"], svd_linear)

        tokens = layername.split('.')
        if len(tokens) == 5 and tokens[2].isnumeric():
            rates[tokens[2]][tokens[-1]] = rank

    with open('rates.json', 'w') as f:
        json.dump(rates, f)


def fixed_truncation_rank(model, compression_ratio, args):
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    for raw_linear in tqdm(linear_info, total=len(module_dict)):
        # set ratio
        info = linear_info[raw_linear]

        svd_linear = SVDLinear.from_linear(
            raw_linear,
            param_ratio=compression_ratio,
            alpha=args.alpha,
            act_aware=args.act_aware,
            sigma_fuse=args.sigma_fuse,
        )
        setattr(info["father"], info["name"], svd_linear)
        # print(f'Using ratio: {ratio:0.4f} for layer: {info["name"]}')

        
