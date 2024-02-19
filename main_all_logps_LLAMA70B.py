import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import datasets
import transformers
import re
from evaluate import load
import random

from gpytorch import settings
from gpytorch.models.exact_prediction_strategies import prediction_strategy
from linear_operator import to_linear_operator
import bert_score
# bertscore = load("bertscore")


import matplotlib.pyplot as plt
import numpy as np
import datasets
import transformers
import re
import torch
import torch.nn.functional as F
import tqdm
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import argparse
import datetime
import os
import json
import functools
import custom_datasets
from multiprocessing.pool import ThreadPool
import time

dis_dict = {}

# 15 colorblind-friendly colors
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
            "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
            "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")


def load_base_model():
    # from accelerate import Accelerator  
  
    # accelerator = Accelerator()  
    # device = accelerator.device 
    print('MOVING BASE MODEL TO GPU...', end='', flush=True)
    start = time.time()
    try:
        mask_model.cpu()
    except NameError:
        pass
    if args.openai_model is None:
        # DEVICE="cuda:0"
        if args.base_model_name == "meta-llama/Llama-2-70b-hf":
            pass
        else:
            base_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')


def load_mask_model():
    print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    start = time.time()

    if args.openai_model is None:
        if args.base_model_name == "meta-llama/Llama-2-70b-hf":
            pass
        else:
            base_model.cpu()
    if not args.random_fills:
        # DEVICE = "cuda:1"
        mask_model.to("cuda:3")
    print(f'DONE ({time.time() - start:.2f}s)')


def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts):
    n_expected = count_masks(texts)
    # mask_model.to("cuda:1")
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to("cuda:3")#gaicheng gen mask yiyang
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=args.mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(texts, span_length, pct, ceil_pct=False):
    if not args.random_fills:
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
        raw_fills = replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = replace_masks(masked_texts)
            extracted_fills = extract_fills(raw_fills)
            new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
    else:
        if args.random_fills_tokens:
            # tokenize base_tokenizer
            tokens = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
            valid_tokens = tokens.input_ids != base_tokenizer.pad_token_id
            replace_pct = args.pct_words_masked * (args.span_length / (args.span_length + 2 * args.buffer_size))

            # replace replace_pct of input_ids with random tokens
            random_mask = torch.rand(tokens.input_ids.shape, device=DEVICE) < replace_pct
            random_mask &= valid_tokens
            random_tokens = torch.randint(0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            # while any of the random tokens are special tokens, replace them with random non-special tokens
            while any(base_tokenizer.decode(x) in base_tokenizer.all_special_tokens for x in random_tokens):
                random_tokens = torch.randint(0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            tokens.input_ids[random_mask] = random_tokens
            perturbed_texts = base_tokenizer.batch_decode(tokens.input_ids, skip_special_tokens=True)
        else:
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
            perturbed_texts = masked_texts
            # replace each <extra_id_*> with args.span_length random words from FILL_DICTIONARY
            for idx, text in enumerate(perturbed_texts):
                filled_text = text
                for fill_idx in range(count_masks([text])[0]):
                    fill = random.sample(FILL_DICTIONARY, span_length)
                    filled_text = filled_text.replace(f"<extra_id_{fill_idx}>", " ".join(fill))
                assert count_masks([filled_text])[0] == 0, "Failed to replace all masks"
                perturbed_texts[idx] = filled_text

    return perturbed_texts


# def perturb_texts(texts, span_length, pct, ceil_pct=False):
#     chunk_size = args.chunk_size
#     if '11b' in mask_filling_model_name:
#         chunk_size //= 2

#     outputs = []
#     for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
#         outputs.extend(perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
#     return outputs

def gpt2_logp(text):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
    # return -base_model(**tokenized, labels=labels).loss.item()
    return -base_model(**{k: v for k, v in tokenized.items() if k in ['input_ids', 'attention_mask']}, labels=labels).loss.item()



def t5_modify(texts, span_length, pct, ceil_pct=False): 
    # print(texts)    
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
    # print(masked_texts)
    raw_fills = replace_masks(masked_texts)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1

    return perturbed_texts

class CustomKernel(gpytorch.kernels.kernel.Kernel):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        print(self.has_lengthscale)
        self.register_parameter(name='hyperparameter1', parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1))) 
        self.register_parameter(name='hyperparameter2', parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1)))

    def forward(self, x1, x2, diag=False, **params):
        dist = []

        for prediction in x1:
            long_dist = []
            for reference in x2:
                long_dist.append(dis_dict[prediction+reference])
            dist.append(long_dist)

        return torch.tensor(dist,dtype=torch.float32) * torch.exp(self.hyperparameter2) + self.hyperparameter1
    
    # def forward(self, x1, x2, diag=False, **params):
    #     # x1, x2 are two lists
    #     x1_ = torch.stack(x1).div(self.hyperparameter1).unsqueeze(-1)
    #     x2_ = torch.stack(x2).div(self.hyperparameter1).unsqueeze(-1)
    #     return  x1_ @ x2_.T * self.hyperparameter2

    def __call__(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        res = to_linear_operator(
            self.forward(x1, x2)
        )
        return res

    def hyperparameters(self,):
        return ', '.join(["%.3f" % param.item() for param in [self.hyperparameter1, self.hyperparameter2]])

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, likelihood):
        super(ExactGPModel, self).__init__(None, None, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = CustomKernel()

    def forward(self, x):
        # x is a list
        # print("------")
        # print(x)
        mean_x = self.mean_module(torch.zeros(len(x)).to(DEVICE))
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, *args, **kwargs):
        if self.training:
            return self.forward(*args, **kwargs)
        else:
            train_inputs = args[0]
            train_targets = args[1]
            inputs = args[2]
            # Get the terms that only depend on training data
            if self.prediction_strategy is None:
                train_output = self.forward(*[train_inputs], **kwargs)

                # Create the prediction strategy for
                self.prediction_strategy = prediction_strategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    # train_labels=torch.stack(train_targets),
                    train_labels=torch.tensor(train_targets),
                    likelihood=self.likelihood,
                )

            # Concatenate the input to the training input
            if isinstance(train_inputs, list):
                # print(train_inputs)
                # print(inputs)
                full_inputs = train_inputs + inputs                
            else:
                full_inputs = torch.cat([train_inputs, inputs], dim=0)

            # Get the joint distribution for training/test data
            full_output = self.forward(*[full_inputs], **kwargs)
            full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

            # Determine the shape of the joint distribution
            batch_shape = full_output.batch_shape
            joint_shape = full_output.event_shape
            tasks_shape = joint_shape[1:]  # For multitask learning
            test_shape = torch.Size([joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape])

            # Make the prediction
            with settings.cg_tolerance(settings.eval_cg_tolerance.value()):
                predictive_mean, predictive_covar = self.prediction_strategy.exact_prediction(full_mean, full_covar)

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
            return full_output.__class__(predictive_mean, predictive_covar)

class GPR:
    def __init__(self):
       self.model = None
       self.likelihood = None

    def fit(self, train_x, train_y, training_iter=50):
        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(likelihood)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, training_iter)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            # print(output)
            # print(train_y)
            # train_y = [-3.6573283672332764, -3.6573283672332764]
            # torch.stack(train_y) = tensor([-3.6573, -3.6573])
        
            loss = -mll(output, torch.tensor(train_y))
            loss.backward()
            # print('Iter %d/%d - Loss: %.3f   hyperparameters: %s   noise: %.3f' % (
            #     i + 1, training_iter, loss.item(),
            #     model.covar_module.hyperparameters(),
            #     model.likelihood.noise.item()
            # ))
            optimizer.step()
            # scheduler.step()
        
        self.model = model
        self.likelihood = likelihood

    def infer(self, train_x, train_y, test_x):
        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()

        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            normal_dist = self.model(train_x, train_y, test_x)
            observed_pred = self.likelihood(normal_dist)
        return normal_dist.covariance_matrix, observed_pred.mean, observed_pred.confidence_region()

def unit_test_of_gpr():
    train_x = torch.linspace(0, 1, 15)
    train_y = torch.sin(train_x * (2 * math.pi))

    train_x = list(train_x)
    train_y = list(train_y)
    
    gpr = GPR()
    gpr.fit(train_x, train_y)

    with torch.no_grad():
        test_x = list(torch.linspace(0, 5, 51))
        gram, pred_mean, (lower, upper) = gpr.infer(train_x, train_y, test_x)

        # Initialize plot
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Plot training data as black stars
        ax.plot([it.item() for it in train_x], [it.item() for it in train_y], 'k*')
        # Plot predictive means as blue line
        ax.plot([it.item() for it in test_x], pred_mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between([it.item() for it in test_x], lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        fig.savefig("gpr.pdf")

        print("uncertainties", gram.diagonal(dim1=-2, dim2=-1).numpy())

class BayesianOpt:
    def __init__(self, x, logp_func, n_perturbation, random_perturb_func,span_length, pct, ceil_pct):
        
        self.logp_func = logp_func
        self.n_perturbation = n_perturbation
        self.all_logps={}
        load_mask_model()
        half_n = len(x)//2
        self.hat_all = random_perturb_func(x[:half_n], span_length, pct, ceil_pct)
        self.hat_all.extend(random_perturb_func(x[half_n:], span_length, pct, ceil_pct))
        hat_all_copy = self.hat_all.copy()
        self.hat_all_backup = self.hat_all.copy()
        hat_all_copy.append(x[0])
        #清空字典
        dis_dict.clear()
        predictions = []
        references = []
        for prediction in hat_all_copy:
            for reference in hat_all_copy:
                predictions.append(prediction)
                references.append(reference)
        half = len(predictions)//2 
        _, _, F1 = bert_score.score(predictions[:half], references[:half], lang='en', verbose=True, device=DEVICE)
        long_dist = F1.cpu().tolist()
        _, _, F1 = bert_score.score(predictions[half:], references[half:], lang='en', verbose=True, device=DEVICE)
        long_dist.extend(F1.cpu().tolist())
        # long_dist = bertscore.compute(predictions=predictions, references=references, lang="en")['f1']
        count = 0
        for prediction in hat_all_copy:
            for reference in hat_all_copy:
                dis_dict[prediction+reference] = long_dist[count]
                count += 1
        load_base_model()
        # print(x)
        # print(x[0])
        # print(self.hat_all)
        self.x = x[0]
        x_logp = logp_func(self.x)
        # randomly perturb once
        random_num = random.randint(0, len(self.hat_all)-1)
        x_hat = self.hat_all[random_num]
        x_hat_logp = logp_func(x_hat)
        self.hat_all.pop(random_num)

        self.samples = [self.x, x_hat]
        self.targets = [x_logp, x_hat_logp]

        self.gpr = GPR()
        self.gpr.fit(self.samples, self.targets)
    
    def main_loop(self):
        while len(self.samples) - 1 < self.n_perturbation + 1:
            with torch.no_grad():
                self.all_logps[len(self.samples)-1] = self.gpr.infer(self.samples, self.targets, self.hat_all_backup)[1].tolist()
            next_x = self.maximum_uncertainty(self.samples, self.targets)
            next_x_logp = self.logp_func(next_x)
            self.samples.append(next_x)
            self.targets.append(next_x_logp)
            self.gpr.fit(self.samples, self.targets)
            # print("self.all_logps:",self.all_logps)
        return
    
    def uncertainty_fn(self, train_x, train_y, x):
        return self.gpr.infer(train_x, train_y, x)[0].diagonal(dim1=-2, dim2=-1)

    def maximum_uncertainty(self, train_x, train_y):
        '''
        todo: 
        find a text next_x that maximizes self.uncertaunty_fn(train_x, train_y, next_x) around the original text self.samples[0] (just like attack on texts)
        '''
        uncertainties = self.uncertainty_fn(train_x, train_y, self.hat_all)
        # 寻找tensor中最大元素的索引
        index = torch.argmax(uncertainties)
        # 去掉一个元素
        next_x = self.hat_all[index]
        self.hat_all.pop(index)
        
        return next_x

def perturb_texts(x, span_length, pct, ceil_pct=False,n_perturbation=1):
    logp_func = gpt2_logp
    random_perturb_func = t5_modify
    # bo = BayesianOpt([x]*250, logp_func, n_perturbation, random_perturb_func, span_length, pct, ceil_pct)
    bo = BayesianOpt([x]*150, logp_func, n_perturbation, random_perturb_func, span_length, pct, ceil_pct) #2.12 modified
    bo.main_loop()
    # bo.all_logps

    # ret_texts = bo.samples
    # ret_texts.pop(0)

    # for i in range(10):
    # with torch.no_grad():
    #     all_logp = bo.gpr.infer(bo.samples, bo.targets, bo.hat_all_backup)[1]
    return bo.all_logps



def drop_last_word(text):
    return ' '.join(text.split(' ')[:-1])


def _openai_sample(p):
    if args.dataset != 'pubmed':  # keep Answer: prefix for pubmed
        p = drop_last_word(p)

    # sample from the openai model
    kwargs = { "engine": args.openai_model, "max_tokens": 200 }
    if args.do_top_p:
        kwargs['top_p'] = args.top_p
    
    r = openai.Completion.create(prompt=f"{p}", **kwargs)
    return p + r['choices'][0].text


# sample from base_model using ****only**** the first 30 tokens in each example as context
def sample_from_model(texts, min_words=55, prompt_tokens=30):
    # encode each text as a list of token ids
    if args.dataset == 'pubmed':
        texts = [t[:t.index(custom_datasets.SEPARATOR)] for t in texts]
        all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    else:
        all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
        all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}
    # print(all_encoded)
    if args.openai_model:
        # decode the prefixes back in to text
        prefixes = base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
        pool = ThreadPool(args.batch_size)

        decoded = pool.map(_openai_sample, prefixes)
    else:
        decoded = ['' for _ in range(len(texts))]

        # sample from the model until we get a sample with at least min_words words for each example
        # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
        tries = 0
        while (m := min(len(x.split()) for x in decoded)) < min_words:
            if tries != 0:
                print()
                print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

            sampling_kwargs = {}
            if args.do_top_p:
                sampling_kwargs['top_p'] = args.top_p
            elif args.do_top_k:
                sampling_kwargs['top_k'] = args.top_k
            min_length = 50 if args.dataset in ['pubmed'] else 150
            # print(**all_encoded)
            outputs = base_model.generate(all_encoded['input_ids'].cuda(), min_length=min_length, max_length=200, do_sample=True, **sampling_kwargs, pad_token_id=base_tokenizer.eos_token_id, eos_token_id=base_tokenizer.eos_token_id)
            decoded = base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            tries += 1

    if args.openai_model:
        global API_TOKEN_COUNTER

        # count total number of tokens with GPT2_TOKENIZER
        total_tokens = sum(len(GPT2_TOKENIZER.encode(x)) for x in decoded)
        API_TOKEN_COUNTER += total_tokens

    return decoded


def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])[:-1]
    labels = labels.view(-1)[1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean()


# Get the log likelihood of each text under the base_model
def get_ll(text):
    if args.openai_model:        
        kwargs = { "engine": args.openai_model, "temperature": 0, "max_tokens": 0, "echo": True, "logprobs": 0}
        r = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
        result = r['choices'][0]
        tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

        assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

        return np.mean(logprobs)
    else:
        with torch.no_grad():
            tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
            labels = tokenized.input_ids
            # return -base_model(**tokenized, labels=labels).loss.item()
            return -base_model(**{k: v for k, v in tokenized.items() if k in ['input_ids', 'attention_mask']}, labels=labels).loss.item()


def get_lls(texts):
    if not args.openai_model:
        return [get_ll(text) for text in texts]
    else:
        global API_TOKEN_COUNTER

        # use GPT2_TOKENIZER to get total number of tokens
        total_tokens = sum(len(GPT2_TOKENIZER.encode(text)) for text in texts)
        API_TOKEN_COUNTER += total_tokens * 2  # multiply by two because OpenAI double-counts echo_prompt tokens

        pool = ThreadPool(args.batch_size)
        return pool.map(get_ll, texts)


# get the average rank of each observed token sorted by model likelihood
def get_rank(text, log=False):
    assert args.openai_model is None, "get_rank not implemented for OpenAI models"

    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:,:-1]
        labels = tokenized.input_ids[:,1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:,-1], matches[:,-2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1 # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()


# get average entropy of each token in the text
def get_entropy(text):
    assert args.openai_model is None, "get_entropy not implemented for OpenAI models"

    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:,:-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()


def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)


# save the ROC curve for each experiment, given a list of output dictionaries, one for each experiment, using colorblind-friendly colors
def save_roc_curves(experiments):
    # first, clear plt
    plt.clf()

    for experiment, color in zip(experiments, COLORS):
        metrics = experiment["metrics"]
        plt.plot(metrics["fpr"], metrics["tpr"], label=f"{experiment['name']}, roc_auc={metrics['roc_auc']:.3f}", color=color)
        # print roc_auc for this experiment
        print(f"{experiment['name']} roc_auc: {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves ({base_model_name} - {args.mask_filling_model_name})')
    plt.legend(loc="lower right", fontsize=6)
    plt.savefig(f"{SAVE_FOLDER}/roc_curves.png")


# save the histogram of log likelihoods in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed
def save_ll_histograms(experiments):
    # first, clear plt
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)
            plt.hist([r["sampled_ll"] for r in results], alpha=0.5, bins='auto', label='sampled')
            plt.hist([r["perturbed_sampled_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed sampled')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.subplot(1, 2, 2)
            plt.hist([r["original_ll"] for r in results], alpha=0.5, bins='auto', label='original')
            plt.hist([r["perturbed_original_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed original')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(f"{SAVE_FOLDER}/ll_histograms_{experiment['name']}.png")
        except:
            pass


# save the histograms of log likelihood ratios in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed
def save_llr_histograms(experiments):
    # first, clear plt
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)

            # compute the log likelihood ratio for each result
            for r in results:
                r["sampled_llr"] = r["sampled_ll"] - r["perturbed_sampled_ll"]
                r["original_llr"] = r["original_ll"] - r["perturbed_original_ll"]
            
            plt.hist([r["sampled_llr"] for r in results], alpha=0.5, bins='auto', label='sampled')
            plt.hist([r["original_llr"] for r in results], alpha=0.5, bins='auto', label='original')
            plt.xlabel("log likelihood ratio")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(f"{SAVE_FOLDER}/llr_histograms_{experiment['name']}.png")
        except:
            pass


def get_perturbation_results(span_length=10, n_perturbations=1, n_samples=500):
    # load_mask_model()

    torch.manual_seed(2)
    np.random.seed(1)

    results = []
    original_text = data["original"]
    sampled_text = data["sampled"]

    # load_base_model()

    perturb_fn = functools.partial(perturb_texts, span_length=span_length, pct=args.pct_words_masked,n_perturbation=n_perturbations)

    p_sampled_text = []
    for x in sampled_text:
        p_sampled_text.append(perturb_fn(x))

    p_original_text = []
    for x in original_text:
        p_original_text.append(perturb_fn(x))

    load_base_model()
    # p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(n_perturbations)])
    # p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])
    # for _ in range(n_perturbation_rounds - 1): 
    #     try:
    #         p_sampled_text, p_original_text = perturb_fn(p_sampled_text), perturb_fn(p_original_text)
    #     except AssertionError:
    #         break
    
    n_generation = 1
    print(len(p_sampled_text),len(sampled_text), n_generation)
    assert len(p_sampled_text) == len(sampled_text) * n_generation, f"Expected {len(sampled_text) * n_generation} perturbed samples, got {len(p_sampled_text)}"
    assert len(p_original_text) == len(original_text) * n_generation, f"Expected {len(original_text) * n_generation} perturbed samples, got {len(p_original_text)}"

    results_dict = {}
    for big_idx in range(10):
        results = []
        for idx in range(len(original_text)):
            results.append({
                "original": original_text[idx],
                "sampled": sampled_text[idx],
                "perturbed_sampled": p_sampled_text[idx][big_idx+1],
                "perturbed_original": p_original_text[idx][big_idx+1]
            })
        results_dict[big_idx+1]=results

    import copy
    for big_idx in range(10):
        results = copy.deepcopy(results_dict[big_idx+1])
        for res in tqdm.tqdm(results, desc="Computing log likelihoods"):
            p_sampled_ll = res["perturbed_sampled"]
            p_original_ll = res["perturbed_original"]
            res["original_ll"] = get_ll(res["original"])
            res["sampled_ll"] = get_ll(res["sampled"])
            res["all_perturbed_sampled_ll"] = p_sampled_ll
            res["all_perturbed_original_ll"] = p_original_ll
            res["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
            res["perturbed_original_ll"] = np.mean(p_original_ll)
            res["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1
            res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1
        results_dict[big_idx+1]=copy.deepcopy(results)
    return results_dict


def run_perturbation_experiment(results, criterion, span_length=10, n_perturbations=1, n_samples=500):
    # compute diffs with perturbed
    predictions = {'real': [], 'samples': []}
    for res in results:
        if criterion == 'd':
            predictions['real'].append(res['original_ll'] - res['perturbed_original_ll'])
            predictions['samples'].append(res['sampled_ll'] - res['perturbed_sampled_ll'])
        elif criterion == 'z':
            if res['perturbed_original_ll_std'] == 0:
                res['perturbed_original_ll_std'] = 1
                print("WARNING: std of perturbed original is 0, setting to 1")
                print(f"Number of unique perturbed original texts: {len(set(res['perturbed_original']))}")
                print(f"Original text: {res['original']}")
            if res['perturbed_sampled_ll_std'] == 0:
                res['perturbed_sampled_ll_std'] = 1
                print("WARNING: std of perturbed sampled is 0, setting to 1")
                print(f"Number of unique perturbed sampled texts: {len(set(res['perturbed_sampled']))}")
                print(f"Sampled text: {res['sampled']}")
            predictions['real'].append((res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])
            predictions['samples'].append((res['sampled_ll'] - res['perturbed_sampled_ll']) / res['perturbed_sampled_ll_std'])

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    name = f'perturbation_{n_perturbations}_{criterion}'
    print(f"{name} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'name': name,
        'predictions': predictions,
        'info': {
            'pct_words_masked': args.pct_words_masked,
            'span_length': span_length,
            'n_perturbations': n_perturbations,
            'n_samples': n_samples,
        },
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }


def run_baseline_threshold_experiment(criterion_fn, name, n_samples=500):
    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    for batch in tqdm.tqdm(range(n_samples // batch_size), desc=f"Computing {name} criterion"):
        original_text = data["original"][batch * batch_size:(batch + 1) * batch_size]
        sampled_text = data["sampled"][batch * batch_size:(batch + 1) * batch_size]

        for idx in range(len(original_text)):
            results.append({
                "original": original_text[idx],
                "original_crit": criterion_fn(original_text[idx]),
                "sampled": sampled_text[idx],
                "sampled_crit": criterion_fn(sampled_text[idx]),
            })

    # compute prediction scores for real/sampled passages
    predictions = {
        'real': [x["original_crit"] for x in results],
        'samples': [x["sampled_crit"] for x in results],
    }

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"{name}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'name': f'{name}_threshold',
        'predictions': predictions,
        'info': {
            'n_samples': n_samples,
        },
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }


# strip newlines from each example; replace one or more newlines with a single space
def strip_newlines(text):
    return ' '.join(text.split())


# trim to shorter length
def trim_to_shorter_length(texta, textb):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
    texta = ' '.join(texta.split(' ')[:shorter_length])
    textb = ' '.join(textb.split(' ')[:shorter_length])
    return texta, textb


def truncate_to_substring(text, substring, idx_occurrence):
    # truncate everything after the idx_occurrence occurrence of substring
    assert idx_occurrence > 0, 'idx_occurrence must be > 0'
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]


def generate_samples(raw_data, batch_size):
    torch.manual_seed(42)
    np.random.seed(42)
    data = {
        "original": [],
        "sampled": [],
    }

    for batch in range(len(raw_data) // batch_size):
        print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
        original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
        sampled_text = sample_from_model(original_text, min_words=30 if args.dataset in ['pubmed'] else 55)

        for o, s in zip(original_text, sampled_text):
            if args.dataset == 'pubmed':
                s = truncate_to_substring(s, 'Question:', 2)
                o = o.replace(custom_datasets.SEPARATOR, ' ')

            o, s = trim_to_shorter_length(o, s)

            # add to the data
            data["original"].append(o)
            data["sampled"].append(s)
    
    if args.pre_perturb_pct > 0:
        print(f'APPLYING {args.pre_perturb_pct}, {args.pre_perturb_span_length} PRE-PERTURBATIONS')
        load_mask_model()
        data["sampled"] = perturb_texts(data["sampled"], args.pre_perturb_span_length, args.pre_perturb_pct, ceil_pct=True)
        load_base_model()

    return data


def generate_data(dataset, key):
    # load data
    if dataset in custom_datasets.DATASETS:
        data = custom_datasets.load(dataset, cache_dir)
    else:
        data = datasets.load_dataset(dataset, split='train', cache_dir=cache_dir)[key]

    # get unique examples, strip whitespace, and remove newlines
    # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
    # then take just the examples that are <= 512 tokens (for the mask model)
    # then generate n_samples samples

    # remove duplicates from the data
    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

    # strip whitespace around each example
    data = [x.strip() for x in data]

    # remove newlines from each example
    data = [strip_newlines(x) for x in data]

    # try to keep only examples with > 250 words
    if dataset in ['writing', 'squad', 'xsum']:
        long_data = [x for x in data if len(x.split()) > 250]
        if len(long_data) > 0:
            data = long_data

    random.seed(0)
    random.shuffle(data)

    data = data[:5_000]

    # keep only examples with <= 512 tokens according to mask_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    tokenized_data = preproc_tokenizer(data)
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]

    # print stats about remainining data
    print(f"Total number of samples: {len(data)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

    return generate_samples(data[:n_samples], batch_size=batch_size)


# def load_base_model_and_tokenizer(name):
#     if args.openai_model is None:
#         print(f'Loading BASE model {args.base_model_name}...')
#         base_model_kwargs = {}
#         if 'gpt-j' in name or 'neox' in name:
#             base_model_kwargs.update(dict(torch_dtype=torch.float16))
#         if 'gpt-j' in name:
#             base_model_kwargs.update(dict(revision='float16'))
#         base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, cache_dir=cache_dir)
#     else:
#         base_model = None

#     optional_tok_kwargs = {}
#     if "facebook/opt-" in name:
#         print("Using non-fast tokenizer for OPT")
#         optional_tok_kwargs['fast'] = False
#     if args.dataset in ['pubmed']:
#         optional_tok_kwargs['padding_side'] = 'left'
#     base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=cache_dir)
#     base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

#     return base_model, base_tokenizer
def load_base_model_and_tokenizer(name):
    if args.openai_model is None:
        print(f'Loading BASE model {args.base_model_name}...')
        base_model_kwargs = {}
        if 'gpt-j' in name or 'neox' in name:
            base_model_kwargs.update(dict(torch_dtype=torch.float16))
        if 'gpt-j' in name:
            base_model_kwargs.update(dict(revision='float16'))
        print(name)
        if name=="meta-llama/Llama-2-70b-hf":
            print(70)
            base_model = transformers.AutoModelForCausalLM.from_pretrained(name, cache_dir=cache_dir,use_auth_token="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",trust_remote_code=True, load_in_4bit=True,torch_dtype=torch.float16, device_map="auto")
        else:
            base_model = transformers.AutoModelForCausalLM.from_pretrained(name, cache_dir=cache_dir,use_auth_token="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",trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")

        # model_path = "meta-llama/Llama-2-7b-chat-hf"
# tokenizer = AutoTokenizer.from_pretrained(model_path,use_auth_token="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",trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_path,use_auth_token="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",trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    else:
        base_model = None

    optional_tok_kwargs = {}
    if "facebook/opt-" in name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if args.dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    print(2)
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, cache_dir=cache_dir,use_auth_token="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",trust_remote_code=True)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    return base_model, base_tokenizer

def eval_supervised(data, model):
    print(f'Beginning supervised evaluation with {model}...')
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(model, cache_dir=cache_dir).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)

    real, fake = data['original'], data['sampled']

    with torch.no_grad():
        # get predictions for real
        real_preds = []
        for batch in tqdm.tqdm(range(len(real) // batch_size), desc="Evaluating real"):
            batch_real = real[batch * batch_size:(batch + 1) * batch_size]
            batch_real = tokenizer(batch_real, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            real_preds.extend(detector(**batch_real).logits.softmax(-1)[:,0].tolist())
        
        # get predictions for fake
        fake_preds = []
        for batch in tqdm.tqdm(range(len(fake) // batch_size), desc="Evaluating fake"):
            batch_fake = fake[batch * batch_size:(batch + 1) * batch_size]
            batch_fake = tokenizer(batch_fake, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            fake_preds.extend(detector(**batch_fake).logits.softmax(-1)[:,0].tolist())

    predictions = {
        'real': real_preds,
        'samples': fake_preds,
    }

    fpr, tpr, roc_auc = get_roc_metrics(real_preds, fake_preds)
    p, r, pr_auc = get_precision_recall_metrics(real_preds, fake_preds)
    print(f"{model} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")

    # free GPU memory
    del detector
    torch.cuda.empty_cache()

    return {
        'name': model,
        'predictions': predictions,
        'info': {
            'n_samples': n_samples,
        },
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }


if __name__ == '__main__':
    DEVICE = "cuda:0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_key', type=str, default="document")
    parser.add_argument('--pct_words_masked', type=float, default=0.3) # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--n_perturbation_list', type=str, default="1,10")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--scoring_model_name', type=str, default="")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-large")
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--base_half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--output_name', type=str, default="")
    parser.add_argument('--openai_model', type=str, default=None)
    parser.add_argument('--openai_key', type=str)
    parser.add_argument('--baselines_only', action='store_true')
    parser.add_argument('--skip_baselines', action='store_true')
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--pre_perturb_pct', type=float, default=0.0)
    parser.add_argument('--pre_perturb_span_length', type=int, default=5)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--random_fills_tokens', action='store_true')
    parser.add_argument('--cache_dir', type=str, default="/home/yibo/.cache")
    args = parser.parse_args()

    API_TOKEN_COUNTER = 0

    if args.openai_model is not None:
        import openai
        assert args.openai_key is not None, "Must provide OpenAI API key as --openai_key"
        openai.api_key = args.openai_key

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

    # define SAVE_FOLDER as the timestamp - base model name - mask filling model name
    # create it if it doesn't exist
    precision_string = "int8" if args.int8 else ("fp16" if args.half else "fp32")
    sampling_string = "top_k" if args.do_top_k else ("top_p" if args.do_top_p else "temp")
    output_subfolder = f"{args.output_name}/" if args.output_name else ""
    if args.openai_model is None:
        base_model_name = args.base_model_name.replace('/', '_')
    else:
        base_model_name = "openai-" + args.openai_model.replace('/', '_')
    scoring_model_string = (f"-{args.scoring_model_name}" if args.scoring_model_name else "").replace('/', '_')
    SAVE_FOLDER = f"tmp_results/{output_subfolder}{base_model_name}{scoring_model_string}-{args.mask_filling_model_name}-{sampling_string}/{START_DATE}-{START_TIME}-{precision_string}-{args.pct_words_masked}-{args.n_perturbation_rounds}-{args.dataset}-{args.n_samples}"
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_FOLDER)}")

    # write args to file
    with open(os.path.join(SAVE_FOLDER, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    mask_filling_model_name = args.mask_filling_model_name
    n_samples = args.n_samples
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    n_perturbation_rounds = args.n_perturbation_rounds
    n_similarity_samples = args.n_similarity_samples

    cache_dir = args.cache_dir
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)

    # generic generative model
    base_model, base_tokenizer = load_base_model_and_tokenizer(args.base_model_name)

    # mask filling t5 model
    if not args.baselines_only and not args.random_fills:
        int8_kwargs = {}
        half_kwargs = {}
        if args.int8:
            int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
        elif args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)
        print(f'Loading mask filling model {mask_filling_model_name}...')
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name, **int8_kwargs, **half_kwargs, cache_dir=cache_dir)
        try:
            n_positions = mask_model.config.n_positions
        except AttributeError:
            n_positions = 512
    else:
        n_positions = 512
    preproc_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=512, cache_dir=cache_dir)
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=n_positions, cache_dir=cache_dir)
    if args.dataset in ['english', 'german']:
        preproc_tokenizer = mask_tokenizer

    load_base_model()
    print("memory used:{}".format(torch.cuda.memory_allocated(0)))
    print(f'Loading dataset {args.dataset}...')
    data = generate_data(args.dataset, args.dataset_key)
    if args.random_fills:
        FILL_DICTIONARY = set()
        for texts in data.values():
            for text in texts:
                FILL_DICTIONARY.update(text.split())
        FILL_DICTIONARY = sorted(list(FILL_DICTIONARY))

    if args.scoring_model_name:
        print(f'Loading SCORING model {args.scoring_model_name}...')
        del base_model
        del base_tokenizer
        torch.cuda.empty_cache()
        base_model, base_tokenizer = load_base_model_and_tokenizer(args.scoring_model_name)
        load_base_model()  # Load again because we've deleted/replaced the old model

    # write the data to a json file in the save folder
    with open(os.path.join(SAVE_FOLDER, "raw_data.json"), "w") as f:
        print(f"Writing raw data to {os.path.join(SAVE_FOLDER, 'raw_data.json')}")
        json.dump(data, f)

    # if not args.skip_baselines:
    #     baseline_outputs = [run_baseline_threshold_experiment(get_ll, "likelihood", n_samples=n_samples)]
    #     if args.openai_model is None:
    #         rank_criterion = lambda text: -get_rank(text, log=False)
    #         baseline_outputs.append(run_baseline_threshold_experiment(rank_criterion, "rank", n_samples=n_samples))
    #         logrank_criterion = lambda text: -get_rank(text, log=True)
    #         baseline_outputs.append(run_baseline_threshold_experiment(logrank_criterion, "log_rank", n_samples=n_samples))
    #         entropy_criterion = lambda text: get_entropy(text)
    #         baseline_outputs.append(run_baseline_threshold_experiment(entropy_criterion, "entropy", n_samples=n_samples))

    #     baseline_outputs.append(eval_supervised(data, model='roberta-base-openai-detector'))
    #     baseline_outputs.append(eval_supervised(data, model='roberta-large-openai-detector'))

    outputs = []

    if not args.baselines_only:
        # run perturbation experiments
        for n_perturbations in n_perturbation_list:
            perturbation_results_dict = get_perturbation_results(args.span_length, n_perturbations, n_samples)
            for idx in range(10):
                perturbation_results = perturbation_results_dict[idx+1]
                for perturbation_mode in ['d', 'z']:
                    output = run_perturbation_experiment(
                        perturbation_results, perturbation_mode, span_length=args.span_length, n_perturbations=idx, n_samples=n_samples)
                    outputs.append(output)
                    with open(os.path.join(SAVE_FOLDER, f"perturbation_{idx}_{perturbation_mode}_results.json"), "w") as f:
                        json.dump(output, f)

    # if not args.skip_baselines:
    #     # write likelihood threshold results to a file
    #     with open(os.path.join(SAVE_FOLDER, f"likelihood_threshold_results.json"), "w") as f:
    #         json.dump(baseline_outputs[0], f)

    #     if args.openai_model is None:
    #         # write rank threshold results to a file
    #         with open(os.path.join(SAVE_FOLDER, f"rank_threshold_results.json"), "w") as f:
    #             json.dump(baseline_outputs[1], f)

    #         # write log rank threshold results to a file
    #         with open(os.path.join(SAVE_FOLDER, f"logrank_threshold_results.json"), "w") as f:
    #             json.dump(baseline_outputs[2], f)

    #         # write entropy threshold results to a file
    #         with open(os.path.join(SAVE_FOLDER, f"entropy_threshold_results.json"), "w") as f:
    #             json.dump(baseline_outputs[3], f)
        
    #     # write supervised results to a file
    #     with open(os.path.join(SAVE_FOLDER, f"roberta-base-openai-detector_results.json"), "w") as f:
    #         json.dump(baseline_outputs[-2], f)
        
    #     # write supervised results to a file
    #     with open(os.path.join(SAVE_FOLDER, f"roberta-large-openai-detector_results.json"), "w") as f:
    #         json.dump(baseline_outputs[-1], f)

    #     outputs += baseline_outputs

    save_roc_curves(outputs)
    save_ll_histograms(outputs)
    save_llr_histograms(outputs)

    # move results folder from tmp_results/ to results/, making sure necessary directories exist
    new_folder = SAVE_FOLDER.replace("tmp_results", "results")
    if not os.path.exists(os.path.dirname(new_folder)):
        os.makedirs(os.path.dirname(new_folder))
    os.rename(SAVE_FOLDER, new_folder)

    print(f"Used an *estimated* {API_TOKEN_COUNTER} API tokens (may be inaccurate)")