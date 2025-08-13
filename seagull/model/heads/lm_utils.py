import textwrap
from typing import Optional
import datasets

import torch
import math

from seagull.data_processing.bbpe import BBPETokenizer
from seagull.data_processing.constants import END_OF_CAPTION_TOKEN
from seagull.model.heads.seagull_lm import SeagullLM
from seagull.utils.torch_utils import set_pytorch_backends
from seagull.utils.utils import colored
from seagull.nn.modules.utils.activations import softmax
from torch.utils.data import DataLoader
from seagull.utils.metrics import compute_perplexity

text_wrapper = textwrap.TextWrapper(width=120)
set_pytorch_backends()

@torch.no_grad()
def compute_seagull_perplexity(
    model: SeagullLM,
    dataset: datasets.Dataset,
    batch_size: int,
    ignore_label_idx: int
) -> float:
    #TODO: 7-1
    model.eval()
    device = model.device

    def _pad_collate(batch):
        max_len = max(len(ex["input_ids"]) for ex in batch)
        padded = torch.full(
            (len(batch), max_len),
            fill_value=ignore_label_idx,
            dtype=torch.long,
        )
        for i, ex in enumerate(batch):
            ids = torch.as_tensor(ex["input_ids"], dtype=torch.long)
            padded[i, : ids.size(0)] = ids
        return padded

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_pad_collate,
    )

    total_nll = 0.0
    total_tokens = 0

    for input_ids in loader:
        input_ids = input_ids.to(device)         
        targets  = input_ids[:, 1:].contiguous() 
        context  = input_ids[:, :-1].contiguous() 

        logits, _ = model(context)                
        logits = logits.contiguous()

        
        batch_ppl = compute_perplexity(
            preds=logits,
            labels=targets,
            labels_ignore_idx=ignore_label_idx,
        )

        
        valid = (targets != ignore_label_idx).sum().item()
        batch_nll = math.log(batch_ppl) * valid

        total_nll    += batch_nll
        total_tokens += valid

   
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)



    
    
    

## quick and dirty code, works on cpu
@torch.no_grad()
def get_argmax_predictions(
      model: SeagullLM,
      tokenizer, 
      prompt
):
    model.eval()
    input_ids = torch.tensor([tokenizer.tokenize(prompt)["input_ids"]]).to(model.device)
    logits, _ = model(input_ids)
    argmax_preds = torch.argmax(logits, dim=-1).cpu().numpy()[0]
    outputs_tokenized = [tokenizer.id2token(k) for k in argmax_preds]

    input_ids = input_ids.cpu().numpy()[0]
    inputs_tokenized = [tokenizer.id2token(k) for k in input_ids]
    return inputs_tokenized, outputs_tokenized