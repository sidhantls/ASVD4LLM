import torch
import torch.nn as nn
from tqdm import tqdm
import os

from datautils import get_eval_loaders
# from lm_eval.base import BaseLM
from lm_eval import evaluator
from datasets import load_dataset
import time
import re

from lm_eval.models import huggingface
from lm_eval import simple_evaluate

# deprecated
class BaseLM: 
    def __init__(self):
        pass 

class EvalLM(BaseLM):
    def __init__(
        self,
        model,
        tokenizer,
        # device="cuda:0",
        batch_size=1,
    ):
        super().__init__()

        # assert isinstance(device, str)
        assert isinstance(batch_size, int)

        # self._device = torch.device(device)
        self._device = model.device

        # self.model = model.to(self.device)
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        self.seqlen = 2048

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.model.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0][:, :, :50257]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )


@torch.no_grad()
def evaluate_perplexity(model, dataset, limit):
    """
    dataset: input ids tensor of shape [batch, sequence length]
    """
    nsamples, seqlen = dataset.size()

    nlls = []

    for i in range(nsamples):
        if i == limit:
            break
        input_ids = dataset[i:i+1,:-1].to(model.device)
        labels = dataset[i:i+1,1:].contiguous()
        logits = model(input_ids=input_ids)[0]
        shift_logits = logits[:, :, :]
        shift_labels = labels.to(model.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
    return ppl.item()



@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    model_name,
    tasks,
    eval_ppl="",
    num_fewshot=0,
    limit=-1,
    batch_size=1,
):
    """
    model: model name
    limit: number of test samples for debug, set to -1 is no limit
    tasks: str tasks are split by ,
    num_fewshot: Number of examples in few-shot context
    eval_ppl: str datasets are split by , such as 'wikitext2,ptb,c4'
    """
    lm = EvalLM(model, tokenizer, batch_size=batch_size)
    results = {}
    if eval_ppl:
        for dataset in eval_ppl.split(","):
            cache_testloader = (
                f"/tmp/{dataset}_testloader_{model_name.replace('/', '_')}_all.cache"
            )
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                # print(f"load calibration from {cache_testloader}")
            else:
                testloader = get_eval_loaders(dataset, tokenizer)
                torch.save(testloader, cache_testloader)
            # print(dataset)
            testenc = testloader.input_ids
            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []

            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(
                    lm.device
                )
                outputs = lm.model.model(batch)
                hidden_states = outputs[0]  # .to(lm.model.lm_head.weight.device)
                logits = lm.model.lm_head(hidden_states)  # .contiguous()
                shift_logits = logits[:, :-1, :]  # .contiguous()
                shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][
                    :, 1:
                ].to(lm.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == limit:
                    break
                # if i == 1:
                #     print(
                #         "memory_allocated",
                #         i,
                #         torch.cuda.memory_allocated() / 1024 / 1024,
                #         "max memory_allocated",
                #         torch.cuda.max_memory_allocated() / 1024**2,
                #     )

            ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * lm.seqlen))
            print(dataset, ppl.item())
            lm.model.config.use_cache = use_cache
            # pprint(model)
            results[dataset] = ppl.item()
    if tasks == "mmlu":
        tasks = "hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions"
    if tasks == "llmqat":
        # tasks = "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"
        tasks = "lambada_openai,openbookqa"
    if tasks != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=tasks.split(","),
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=None if limit == -1 else limit,
            no_cache=True,
        )
        t_results = t_results["results"]
        acc_list = [
            t_results[key]["acc"] for key in t_results.keys() if "acc" in t_results[key]
        ]
        t_results["mean"] = sum(acc_list) / len(acc_list)
        results.update(t_results)
        print(results)
        # print mean
        print(f"\n\n===== mean acc: {sum(acc_list)/len(acc_list)} =====\n\n")

    return results


def evaluate_with_harness_full(model, tokenizer, device, debug=False, batch_size=2):
    """
    Evaluates a causall LLM model using evaluation harness on the full dataset, unlike def evaluate_with_harness, 
    which is only on a small susbet 

    Args:
        model (hf model )
        device (str, optional): The device to use for the evaluation ('cpu' or 'cuda'). Default is 'cpu'.
        debug (bool, optional): Whether to run the evaluation in debug mode or not. Default is False.
        batch_size (int, optional): The batch size to use for the evaluation. Default is 2.

    Returns:
        dict: A dictionary containing the evaluation metrics, including the accuracy on the MMLU (MultiModal Lexical Understanding) social sciences task and the exact match accuracy on the Natural Questions (NQ) open-ended task.
    """
    import time

    start = time.time()
    model = model.eval() 
    lm_obj = huggingface.HFLM(pretrained=model, backend='causal', tokenizer=tokenizer, batch_size=batch_size, device=device)

    if debug: 
       limit1 = limit_mmlu = limit_nqopen = 1
    else: 
       limit1, limit_mmlu, limit_nqopen = 1000, 30, 1000
       
    all_metrics = {}

    results1 = simple_evaluate( # call simple_evaluate
            model=lm_obj,
            tasks=["hellaswag", "winogrande", "arc_easy", "arc_challenge", "piqa", "boolq", "openbookqa"],
            num_fewshot=0,
            limit=limit1,
            batch_size=batch_size,
            cache_requests=None,
            log_samples=False,
            bootstrap_iters=0,
            gen_kwargs="max_new_tokens=40",
        )
    
    results_mmlu = simple_evaluate( # call simple_evaluate
        model=lm_obj,
        tasks=['mmlu'],
        num_fewshot=0,
        limit=limit_mmlu,
        device = 'cuda',
        batch_size=batch_size,
        cache_requests=None,
        log_samples=False,
        gen_kwargs="max_new_tokens=40",
        bootstrap_iters=1
    )

    results_nq = simple_evaluate( # call simple_evaluate
        model=lm_obj,
        tasks=['nq_open'],
        num_fewshot=5,
        limit=limit_nqopen,
        device = 'cuda',
        batch_size=batch_size,
        cache_requests=None,
        log_samples=False,
        gen_kwargs="max_new_tokens=40",
        bootstrap_iters=1
    )

    all_metrics = {f'final_eval_harness_shot=0/{key}': results1['results'][key]['acc,none'] for key in results1['results']}
    all_metrics[f'final_eval_harness_shot=5/nq_open'] = results_nq['results']['nq_open']['exact_match,remove_whitespace']
    all_metrics[f'final_eval_harness_shot=0/mmlu'] = results_mmlu['results']['mmlu']['acc,none']

    print(f'Completed evaluation with harness in {time.time()-start: 0.3f} seconds')
    return all_metrics

    
