from typing import Optional, Tuple, Union, List, Any, Set

import torch
import random
import numpy as np
from torch import nn

from seagull.model.seagull_transformer import Seagull
from seagull.nn.modules.glu import GLU
from seagull.nn.modules.linear import Linear
from seagull.nn.modules.module import Module
from seagull.nn.modules.rms_norm import RMSNorm
from seagull.nn.modules.utils.activations import softmax
from seagull.data_processing.constants import END_OF_CAPTION_TOKEN
from seagull.data_processing.bbpe import BBPETokenizer

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def greedy_decode(logits: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Finds next token according to greedy decoding
    
    Parameters
    ----------
    logits: torch.Tensor
        Logits for next token; has shape batch_size x vocab_size
    
    Returns
    -------
    torch.Tensor
        Next token IDs with shape batch_size x 1
    """
    # TODO 4.1-1
    nextTokens = torch.argmax(logits, dim=-1, keepdim=True)
    return nextTokens
def sampling_decode(logits: torch.Tensor, temperature: float, **kwargs) -> torch.Tensor:
    """
    Finds next token according to random sampling
    
    Parameters
    ----------
    logits: torch.Tensor
        Logits for next token; has shape batch_size x vocab_size
    temperature: float
        Sampling temperature
    
    Returns
    -------
    torch.Tensor
        Next token IDs with shape batch_size x 1
    """
    # TODO 4.1-2
    scaledLogits = logits/temperature
    probabilities = softmax(scaledLogits, dim=-1)
    nextTokens = torch.multinomial(probabilities, num_samples=1)

    return nextTokens

def top_k_decode(logits: torch.Tensor, temperature: float, k: float, **kwargs) -> torch.Tensor:
    """
    Finds next token according to top-k decoding
    
    Parameters
    ----------
    logits: torch.Tensor
        Logits for next token; has shape batch_size x vocab_size
    temperature: float
        Sampling temperature
    k: float
        k parameter
    
    Returns
    -------
    torch.Tensor
        Next token IDs with shape batch_size x 1
    """
    # TODO 4.1-3
    scaledLogits = logits/temperature
    topk_vals, topk_indices = torch.topk(scaledLogits, k, dim=-1)

    filtered_logits = scaledLogits.clone()
    mask = torch.ones_like(filtered_logits, dtype=torch.bool)
    mask.scatter_(dim=-1, index=topk_indices, value=False)
    filtered_logits.masked_fill_(mask, float('-inf'))

    probabilities = softmax(filtered_logits, dim=-1)
    nextTokens = torch.multinomial(probabilities, num_samples=1)

    return nextTokens

    
def top_p_decode(logits: torch.Tensor, temperature: float, p: float, **kwargs) -> torch.Tensor:
    """
    Finds next token according to top-p decoding
    
    Parameters
    ----------
    logits: torch.Tensor
        Logits for next token; has shape batch_size x vocab_size
    temperature: float
        Sampling temperature
    p: float
        p parameter
    
    Returns
    -------
    torch.Tensor
        Next token IDs with shape batch_size x 1
    """
    # TODO 4.1-4
    scaledLogits = logits/temperature
    sortedLogits, sortedIndices = torch.sort(scaledLogits, descending=True, dim=-1)
    sortedProbabilities = softmax(sortedLogits, dim=-1)
    cumulative_probs = sortedProbabilities.cumsum(dim=-1)
    
    removed = cumulative_probs>p
    removed[..., 1:] = removed[..., :-1].clone()
    removed[..., 0] = False

    sortedLogits = sortedLogits.masked_fill(removed, float('-inf'))

    filteredLogits = torch.full_like(scaledLogits, float('-inf'))
    filteredLogits.scatter_(dim=-1, index=sortedIndices, src=sortedLogits)
    filtered_probs = softmax(filteredLogits, dim=-1)
    nextTokens = torch.multinomial(filtered_probs, num_samples=1)

    return nextTokens


DECODING_FUNCS = {
    'greedy_decode': greedy_decode,
    'top_p_decode': top_p_decode,
    'top_k_decode': top_k_decode,
    'sampling_decode': sampling_decode
}

def parse_before_eos(all_input_ids: torch.Tensor, eos_token_id: int):
    """
    Parses a batch of input_ids and only maintains the IDs (strictly) before the first EOS token

    Parameters
    ----------
    input_ids: torch.Tensor
        Batched input IDs; has shape batch_size x sequence_length
    eos_token_id: int
        ID of EOS token
    
    Returns
    -------
    List[List[int]]
        Output list of IDs that only maintains the IDs (strictly) before the first EOS token (inclusive)/ or the whole sequence if there is no EOS token. 
        Each item in the List is a List of ints of possibly different lengths.
    """
    # TODO 4.2 -1 
    results = []
    for seq in all_input_ids.tolist():
        if eos_token_id in seq:
            idx = seq.index(eos_token_id)
            results.append(seq[:idx+1])
        else:
            results.append(seq)
    return results

class SeagullLM(Module):
    def __init__(self, weight_tying: bool = True, **seagull_kwargs: Any):
        """
        A language model based on the seagull.model.seagull_transformer.Seagull transformer.

        Parameters:
        weight_tying: bool
        Whether to apply weight tying for the output embeddings.

        seagull_kwargs: Any
        Additional arguments used to initialize a seagull.model.seagull_transformer.Seagull transformer.
        """
        super().__init__()

        self._max_positions = seagull_kwargs["max_positions"]

        self.seagull = Seagull(**seagull_kwargs)
        self.weight_tying = weight_tying
        if not weight_tying:
            self.lm_head = Linear(
                in_features=seagull_kwargs["embedding_dim"],
                out_features=seagull_kwargs["vocab_size"],
                bias=False,
                activation=None,
            )

        self.apply(self._init_weights)

    def reset_kv_cache(self):
        self.seagull.reset_kv_cache()

    def _init_weights(self, module: nn.Module) -> None:
        # Initialize module bias parameters.
        if isinstance(module, (nn.Linear, Linear, RMSNorm, nn.LayerNorm)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, GLU):
            if module.bias_b is not None:
                nn.init.zeros_(module.bias_b)
                nn.init.zeros_(module.bias_c)

        # Initialize module weight parameters.
        if isinstance(module, (nn.Embedding, nn.Linear, Linear)):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, GLU):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.gain)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
        return_output_at_all_layers: bool = False,
        return_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]]:
        """
        Computes a forward training pass through the Seagull transformer.

        Parameters
        ----------
        input_ids : torch.Tensor
            The input IDs passed into the transformer.
        position_ids : Optional[torch.Tensor]
            Optional position IDs to use for positional embeddings.
        padding_mask : Optional[torch.Tensor]
            Optional padding mask indicating which tokens in the embeddings, if any, are padding tokens.
        use_kv_cache : bool
            Whether to use cached keys and values for attention.
        return_output_at_all_layers : bool
            Whether to return the output at all transformer layers, or only the final layer.
        return_attentions : bool
            Whether to return the attentions of all transformer layers in addition to the outputs.

        Returns
        -------
        Tuple[torch.Tensor, Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]]
            A tuple of the output embedding logits and the outputs from the transformer.
        """
        # input_ids, padding_mask: (batch_size, max_length)
        # output_embeddings: (batch_size, max_length, embedding_dim)
        # TODO- copy from HW3
        seagull_out = self.seagull(
            input_ids=input_ids,
            position_ids=position_ids,
            padding_mask=padding_mask,
            use_kv_cache=use_kv_cache,
            return_output_at_all_layers=return_output_at_all_layers,
            return_attentions=return_attentions,
        )
        if return_attentions:                          
            layer_outputs, attentions = seagull_out
        else:                                           
            layer_outputs = seagull_out
            attentions = None
        final_hidden = layer_outputs[-1] if isinstance(layer_outputs, list) else layer_outputs
        if self.weight_tying:
            
            tied_weights = self.seagull.embedding.token_embedding.weight        
            logits = torch.matmul(final_hidden, tied_weights.t())               
        else:
            logits = self.lm_head(final_hidden)                                
        if return_attentions:
            return logits, (layer_outputs, attentions)
        else:
            return logits, layer_outputs        

    @torch.no_grad()
    def talk(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        seed: Optional[int] = 42,
        num_samples: Optional[int] = 1,
        decoding_strategy: Optional[str] = 'greedy_decode',
        decoding_kwargs: Optional[dict] = None,
        eos_token_id: Optional[int] = None,
    ):
        """
        Generates text output by the Seagull language model based on given inputs.

        Parameters
        ----------
        input_ids: torch.Tensor
            The input IDs to feed as input to the language model. Shape 1 x sequence length
        max_new_tokens: int
            The maximum number of tokens to generate.
        seed: Optional[int]
            Seed for reproducibility
        num_samples: Optional[int]
            The number of samples to generate.
        decoding_strategy: Optional[str]
            The name of the decoding strategy
        decoding_kwargs: Optioanl[dict]
            Keyword arguments for the logits -> input ids decoding function
        eos_token_id : Optional[int]
            The token ID corresponding to the end of the text.

        Returns
        -------
        List[List[int]]
            Output list of IDs corresponding to output text generated. Has "dimension" num_samples x (input + output sequence length)
        """
        set_seed(seed)
        batch_size, input_seq_len = input_ids.shape
        all_input_ids = input_ids.unsqueeze(0).expand(num_samples, -1, -1).reshape(-1, input_seq_len)
        all_input_ids = all_input_ids[:, -self._max_positions :].to(self.device)
        all_position_ids = torch.arange(input_seq_len).unsqueeze(0).expand(batch_size, -1).to(self.device)

        self.eval()
        ended_mask = torch.tensor([[False] for _ in range(batch_size * num_samples)]).to(self.device)
        #print(ended_mask.shape)
        for i in range(max_new_tokens):
            lm_logits = self(
                input_ids   = all_input_ids   if i == 0 else all_input_ids[:, -1:],
                position_ids= all_position_ids if i == 0 else all_position_ids[:, -1:],
                use_kv_cache= True
            )[0]
            ##################### START
            # TODO 4.2-2
            next_token_logits = lm_logits[:, -1, :]

            decode_fn = DECODING_FUNCS[decoding_strategy]  
            dk = decoding_kwargs or {}
            next_tokens = decode_fn(next_token_logits, **dk).to(self.device)
            all_input_ids = torch.cat([all_input_ids, next_tokens], dim=1)
            next_pos_ids = all_position_ids[:, -1:] + 1
            all_position_ids = torch.cat([all_position_ids, next_pos_ids], dim=1)
            if eos_token_id is not None:
                ended_mask |= next_tokens.eq(eos_token_id)
                if ended_mask.all():
                    break
            ##################### END
        
        self.reset_kv_cache()  # reset start position to zero
        if eos_token_id:
            return parse_before_eos(all_input_ids, eos_token_id)
        else: ## if eos_token_id is not specified, we do not need to truncate any generation. 
            return all_input_ids.tolist()  
        

def make_seagull_talk(
    prompt: str,
    bbpe_tokenizer: BBPETokenizer,
    seagull_lm: SeagullLM,
    max_new_tokens: int,
    num_samples: int,
    decoding_strategy: Optional[str] = 'greedy_decode',
    decoding_kwargs: Optional[dict] = None
) -> List[str]:
    """
    Generates text using seagull given a prompt

    Parameters
    ----------
    prompt: str
        Prompt
    bbpe_tokenizer: BBPETokenizer
        Tokenizer
    seagull_lm: SeagullLM
        Seagull LM
    max_new_tokens: int
        The maximum number of tokens to generate
    num_samples: int
        The number of samples to generate
    decoding_strategy: Optional[str]
        The name of the decoding strategy
    decoding_kwargs: Optional[dict]
        Keyword arguments for the logits -> input ids decoding function
    """

    if prompt is None or prompt == "":
        prompt = bbpe_tokenizer.eos_token  # generate unconditional samples
    input_ids = torch.tensor(bbpe_tokenizer.encode(text=prompt, add_special_tokens=False), dtype=torch.long)

    completions = seagull_lm.talk(
        input_ids=input_ids.unsqueeze(0),
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
        decoding_strategy=decoding_strategy,
        decoding_kwargs=decoding_kwargs,
        eos_token_id=bbpe_tokenizer.token2id(END_OF_CAPTION_TOKEN)
    )

    return [bbpe_tokenizer.decode(completion) for completion in completions]