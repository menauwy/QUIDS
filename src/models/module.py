import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math
from dataclasses import dataclass
from typing import List, Union
import random

from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    ModelOutput,
    Seq2SeqModelOutput,
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.models.t5.modeling_t5 import (
    T5Stack, 
    T5LayerNorm, 
    T5LayerFF,
    T5Attention, 
    T5LayerSelfAttention, 
    T5LayerCrossAttention,
    T5PreTrainedModel,
    T5Config
)

from transformers.models.bart.modeling_bart import (
    BartAttention,
    BartConfig,
    BartPretrainedModel,
    BartLearnedPositionalEmbedding,
    _make_causal_mask,
    _expand_mask
)
from transformers.activations import ACT2FN

import logging
logger = logging.getLogger(__name__)

@dataclass
class MySeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)` of a tuple of tensors, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state_pos: Optional[torch.FloatTensor] = None
    encoder_hidden_states_pos: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions_pos: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state_neg: Optional[torch.FloatTensor] = None
    encoder_hidden_states_neg: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions_neg: Optional[Tuple[torch.FloatTensor, ...]] = None
    # last_hidden_state_archive: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None

@dataclass
class MyGreedySearchEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions_pos: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states_pos: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions_neg: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states_neg: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

@dataclass
class MyBeamSearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using beam search. Hidden states and attention weights
    of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
    attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`torch.FloatTensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
            of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
            Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
            with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
        beam_indices (`torch.LongTensor`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
            `(batch_size*num_return_sequences, sequence_length)`.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, num_heads, generated_length,
            sequence_length)`.
        cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    encoder_attentions_pos: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states_pos: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions_neg: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states_neg: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

@dataclass
class MyBaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    hidden_states_archive: Optional[Tuple[torch.FloatTensor, ...]] = None

class MyT5Decoder(T5PreTrainedModel):
    """
    a decoder T5Stack
    * use past_key_values, and remove graident checkpointing
    * and hence post_init() changed to init_weights()
    """
    def __init__(self, config, args, embed_tokens=None):
        super().__init__(config)

        self.args = args
        self.embed_tokens = embed_tokens
        self.is_decoder = True

        self.block = nn.ModuleList(
            [MyT5DecoderBlock(config, self.args, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def _get_extended_encoder_attention_mask(self, is_decoder, encoder_hidden_states, encoder_attention_mask, inputs_embeds) -> Tensor:
        # Invert an attention mask (e.g., switches 0. and 1.).
        if is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        return encoder_extended_attention_mask
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states_pos=None,
        encoder_attention_mask_pos=None,
        encoder_hidden_states_neg=None,
        encoder_attention_mask_neg=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None, # False for training
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ): # T5_INPUTS_DOCSTRING contains explanations and sizes for all args
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        # prepare indicators for forward pass and output format
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        ###################### prepare input ######################
        # get input sizes, inputs_embeds to decoder
        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size() # (batch_size, seq_length)
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values with `None` if past does not exist
        ## past_key_values: (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
        ## Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
        ## If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
        ## !!!! don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
        ## !!!! `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        # We can provide a self-attention mask of dimensions [batch_size, seq_length] -> [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # get_extended_attention_mask from modeling_utils.py:  Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable [batch_size, seq_length] to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask_pos = self._get_extended_encoder_attention_mask(
            self.is_decoder, encoder_hidden_states_pos, encoder_attention_mask_pos, inputs_embeds
        )
        encoder_extended_attention_mask_neg = self._get_extended_encoder_attention_mask(
            self.is_decoder, encoder_hidden_states_neg, encoder_attention_mask_neg, inputs_embeds
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None # a tuple of 12 tuple of size 6
        all_hidden_states = () if output_hidden_states else None # a tuple of 13 tuples of size 1
        all_attentions = () if output_attentions else None # a tuple of 12 tuples of size 1
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None # a tuple of 12 tuples of size 2
        all_hidden_states_archive = () # a tuple of 12 tuple of size 3
        position_bias = None #初始化为0
        encoder_decoder_position_bias_pos = None
        encoder_decoder_position_bias_neg = None

        ###################### forward pass ######################
        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states_pos is not None:
                    encoder_hidden_states_pos = encoder_hidden_states_pos.to(hidden_states.device)
                if encoder_extended_attention_mask_pos is not None:
                    encoder_extended_attention_mask_pos = encoder_extended_attention_mask_pos.to(hidden_states.device)
                if encoder_hidden_states_neg is not None:
                    encoder_hidden_states_neg = encoder_hidden_states_neg.to(hidden_states.device)
                if encoder_extended_attention_mask_neg is not None:
                    encoder_extended_attention_mask_neg = encoder_extended_attention_mask_neg.to(hidden_states.device)
                if encoder_decoder_position_bias_pos is not None:
                    encoder_decoder_position_bias_pos = encoder_decoder_position_bias_pos.to(hidden_states.device)
                if encoder_decoder_position_bias_neg is not None:
                    encoder_decoder_position_bias_neg = encoder_decoder_position_bias_neg.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # gradient checkpointing:
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states_pos,
                    encoder_extended_attention_mask_pos,
                    encoder_hidden_states_neg,
                    encoder_extended_attention_mask_neg,
                    encoder_decoder_position_bias_pos,
                    encoder_decoder_position_bias_neg,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
                )
            # the general case: our case that use past_key_value
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states_pos,
                    encoder_extended_attention_mask_pos,
                    encoder_hidden_states_neg,
                    encoder_extended_attention_mask_neg,
                    encoder_decoder_position_bias_pos,
                    encoder_decoder_position_bias_neg,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            ###################### deal with output ######################
            # layer_outputs is a tuple with:
            # (hidden-states, (key-value-states: tuple of 6), self-attention position bias, self-attention weights, 
            # cross-attention position bias_pos, cross-attention weights_pos, cross-attention position bias_neg, cross-attention weights_neg)
            # (hidden_states_archive: tuple of three hidden states)) 1+1*6 + 6 + 1*3
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # !!! We share the position biases between the layers - the first layer store them
            position_bias = layer_outputs[2]
            if self.is_decoder:
                encoder_decoder_position_bias_pos = layer_outputs[4 if output_attentions else 3]
                encoder_decoder_position_bias_neg = layer_outputs[6 if output_attentions else 4]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder: # cross attention for pos + neg
                    _cross_attentions = (layer_outputs[5], layer_outputs[7])
                    all_cross_attentions = all_cross_attentions + (_cross_attentions,)

            # append hidden states archive        
            all_hidden_states_archive = all_hidden_states_archive + (layer_outputs[-1],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                    all_hidden_states_archive,
                ]
                if v is not None
            )
        return MyBaseModelOutputWithPastAndCrossAttentions( # sizes see initialization step
            last_hidden_state=hidden_states, # tensor of size (batch_size, seq_length, dim)
            past_key_values=present_key_value_states, # tuple of (key_states, value_states, k, v, k, v) * num_layers
            hidden_states=all_hidden_states, # tuple of tensors of size (batch_size, seq_length, dim): num_layers + 1
            attentions=all_attentions, # self attention weights of each layer: tuple of tensors of size (batch_size, num_heads, seq_length, seq_length)
            cross_attentions=all_cross_attentions, # cross attention weights of each layer (pos + neg): tuple of tensors of size (batch_size, num_heads, seq_length, seq_length)
            hidden_states_archive=all_hidden_states_archive, # tuple of three hidden states: tuple of tensors of size (batch_size, seq_length, dim)
        )


class MyT5DecoderBlock(nn.Module):
    """
    a decoder T5Block: modified from T5Block
    one self attention layer + two cross attention layers
    
    """
    def __init__(self, config, args, has_relative_attention_bias=False):
        super().__init__()

        self.args = args
        self.d_model = config.d_model
        self.is_decoder = True
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layer = nn.ModuleList()
        self_attn = T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer.append(self_attn) # layer 1: self attention
        if self.is_decoder:
            cross_attn = T5LayerCrossAttention(config) # TODO：原代码不进行position_bias的计算的？
            cross_attn2 = T5LayerCrossAttention(config)
            self.layer.append(cross_attn)
            self.layer.append(cross_attn2)

        self.layer.append(T5LayerFF(config))
        if self.args.model_variant == 'var6' or self.args.model_variant == 'var62': #weighted sum for combination
            self.alpha_self = nn.Parameter(torch.tensor(1.0))
            self.alpha_pos = nn.Parameter(torch.tensor(1.0))
            self.alpha_neg = nn.Parameter(torch.tensor(1.0))
        if self.args.model_variant == 'var7': # concatenation and linear transformation
            self.linear = nn.Linear(2 * config.d_model, config.d_model)
            # self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states_pos=None,
        encoder_attention_mask_pos=None,
        encoder_hidden_states_neg=None,
        encoder_attention_mask_neg=None,
        encoder_decoder_position_bias_pos=None,
        encoder_decoder_position_bias_neg=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False, #True
        output_attentions=False,
        return_dict=True,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states_pos is None else 6 #change 4 to 6

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (key / value) for each cross attention. ' if expected_num_past_key_values == 6 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value_pos = past_key_value[2:4]
            cross_attn_past_key_value_neg = past_key_value[4:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value_pos, cross_attn_past_key_value_neg = None, None, None
        
        # layer 1: self attention
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        For the first layer, self_attn_past_key_value should be None
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states_self, present_key_value_state = self_attention_outputs[:2] # present_key_value_state = (key_states, value_states) 
        
        # clamp inf values to enable fp16 training
        if hidden_states_self.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states_self).any(),
                torch.finfo(hidden_states_self.dtype).max - 1000,
                torch.finfo(hidden_states_self.dtype).max,
            )
            hidden_states_self = torch.clamp(hidden_states_self, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states_pos is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            # query_length: decoder 有用
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            # layer 2: cross attention on relavant doc representations
            cross_attention_outputs_pos = self.layer[1](
                hidden_states_self, # self attention output as query
                key_value_states=encoder_hidden_states_pos, #positive doc encoder output
                attention_mask=encoder_attention_mask_pos,
                position_bias=encoder_decoder_position_bias_pos,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value_pos,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states_pos, cross_attn_present_key_value_pos = cross_attention_outputs_pos[:2]

            # clamp inf values to enable fp16 training
            if hidden_states_pos.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states_pos).any(),
                    torch.finfo(hidden_states_pos.dtype).max - 1000,
                    torch.finfo(hidden_states_pos.dtype).max,
                )
                hidden_states_pos = torch.clamp(hidden_states_pos, min=-clamp_value, max=clamp_value)
            
            # layer 2.2: cross attention on irrelavant doc representations
            cross_attention_outputs_neg = self.layer[2](
                hidden_states_self, # self attention output as query
                key_value_states=encoder_hidden_states_neg, #negative doc encoder output
                attention_mask=encoder_attention_mask_neg,
                position_bias=encoder_decoder_position_bias_neg,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value_neg,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states_neg, cross_attn_present_key_value_neg = cross_attention_outputs_neg[:2]

            # clamp inf values to enable fp16 training
            if hidden_states_neg.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states_neg).any(),
                    torch.finfo(hidden_states_neg.dtype).max - 1000,
                    torch.finfo(hidden_states_neg.dtype).max,
                )
                hidden_states_neg = torch.clamp(hidden_states_neg, min=-clamp_value, max=clamp_value)       

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None: # tuple of (key_states, value_states) * 3
                present_key_value_state = present_key_value_state + cross_attn_present_key_value_pos + cross_attn_present_key_value_neg

            # Keep cross-attention outputs and relative position weights
            attention_outputs = self_attention_outputs[2:] + cross_attention_outputs_pos[2:] + cross_attention_outputs_neg[2:]

            # substract positive hidden states with negative doc hidden states on dim=2
            # (batch_size, seq_length, dim) 不需要加dropout，因为在下一步FFN中会加
            # TODO: 在query_aware的版本下，记得加上self_attention的hidden_states（query）
            # if self.args.model_variant == 'var0':
            #     hidden_states = hidden_states_self +  hidden_states_pos - hidden_states_neg #base
            # elif self.args.model_variant == 'var1':
            #     hidden_states = hidden_states_pos - hidden_states_neg
            # elif self.args.model_variant == 'var2':
            #     hidden_states = hidden_states_self +  0.5 * (hidden_states_pos - hidden_states_neg)
            # elif self.args.model_variant == 'var3':
            #     hidden_states = hidden_states_pos
            # elif self.args.model_variant == 'var4':
            #     hidden_states = hidden_states_self
            # elif self.args.model_variant == 'var5':
            #     hidden_states == hidden_states_self + hidden_states_pos
            # elif self.args.model_variant == 'var6':
            #     hidden_states = self.alpha_self * hidden_states_self + self.alpha_pos * hidden_states_pos - self.alpha_neg * hidden_states_neg
            # elif self.args.model_variant == 'var62':
            #     hidden_states_self = self.alpha_self * hidden_states_self
            #     hidden_states_pos = self.alpha_pos * hidden_states_pos
            #     hidden_states_neg = self.alpha_neg * hidden_states_neg
            #     hidden_states = hidden_states_self + hidden_states_pos - hidden_states_neg
            # elif self.args.model_variant == 'var7':
            #     concatenated_cross_hidden_states = torch.cat((hidden_states_pos, hidden_states_neg), dim=-1)
            #     combined_cross_hidden_states = self.linear(concatenated_cross_hidden_states)
            #     hidden_states = F.relu(hidden_states_self + combined_cross_hidden_states)
            # elif self.args.model_variant == 'var8':
            #     _attention_weights = nn.functional.softmax(torch.cat((hidden_states_pos, hidden_states_neg), dim=-1), dim=-1)
            #     # Weighted sum of cross-attention states
            #     combined_cross_hidden_states = _attention_weights[:, :, :self.d_model] * hidden_states_pos + \
            #                    _attention_weights[:, :, self.d_model:] * hidden_states_neg
            #     hidden_states = hidden_states_self + combined_cross_hidden_states
            # elif self.args.model_variant == 'var9':
            #     scores_pos = torch.matmul(hidden_states_self, hidden_states_pos.transpose(-1, -2))
            #     scaled_scores_pos = scores_pos / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
            #     # _attention_weights_pos = F.softmax(scaled_scores_pos, dim=-1) # (batch, seq, seq)
            #     _attention_weights_pos = self.dropout(F.softmax(scaled_scores_pos, dim=-1))

            #     scores_neg = torch.matmul(hidden_states_self, hidden_states_neg.transpose(-1, -2))
            #     scaled_scores_neg = scores_neg / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
            #     # _attention_weights_neg = F.softmax(scaled_scores_neg, dim=-1)
            #     _attention_weights_neg = self.dropout(F.softmax(scaled_scores_neg, dim=-1))

            #     # Weighted sum of cross-attention states
            #     hidden_states = hidden_states_self + \
            #                     torch.matmul(_attention_weights_pos, hidden_states_pos) - \
            #                     torch.matmul(_attention_weights_neg, hidden_states_neg)
            # elif self.args.model_variant == 'var10':
            hidden_states = self.dropout(hidden_states_self) + self.dropout(hidden_states_pos) - self.dropout(hidden_states_neg)

            # tuple of three hidden states
            hidden_states_archive = (hidden_states_self, hidden_states_pos, hidden_states_neg)

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache: # 1 + 3*2 + 6 = 13
            outputs = outputs + (present_key_value_state,) + attention_outputs + (hidden_states_archive,)
        else:
            outputs = outputs + attention_outputs + (hidden_states_archive,) # make sure hidden_states_achive is a singel tulpe

        return outputs 
        #(hidden-states, (key-value-states: tuple of 6), self-attention position bias, self-attention weights, 
        # cross-attention position bias_pos, cross-attention weights_pos, cross-attention position bias_neg, cross-attention weights_neg,
        # (hidden_states_archive: tuple of three hidden states)) 1+1*6 + 6 + 1*3

class MyBartDecoderBlock(nn.Module):
    def __init__(self, config, args, has_relative_attention_bias=False):
        super().__init__()

        self.args = args
        self.d_model = config.d_model
        self.is_decoder = True
        
        # self attention
        self.self_attn = BartAttention(
            embed_dim=self.d_model,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.d_model)

        # cross attention pos 
        self.encoder_attn_pos = BartAttention(
            self.d_model,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm_pos = nn.LayerNorm(self.d_model)

        # cross attention neg
        self.encoder_attn_neg = BartAttention(
            self.d_model,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm_neg = nn.LayerNorm(self.d_model)

        # feed forward
        self.fc1 = nn.Linear(self.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.d_model)
        self.final_layer_norm = nn.LayerNorm(self.d_model)

        if self.args.model_variant == 'var6':
            self.alpha_self = nn.Parameter(torch.tensor(1.0))
            self.alpha_pos = nn.Parameter(torch.tensor(1.0))
            self.alpha_neg = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states_pos: Optional[torch.Tensor] = None,
        encoder_attention_mask_pos: Optional[torch.Tensor] = None,
        encoder_hidden_states_neg: Optional[torch.Tensor] = None,
        encoder_attention_mask_neg: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, d_model)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, d_model)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # get past key values
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states_pos is None else 6 #change 4 to 6

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (key / value) for each cross attention. ' if expected_num_past_key_values == 6 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value_pos = past_key_value[2:4]
            cross_attn_past_key_value_neg = past_key_value[4:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value_pos, cross_attn_past_key_value_neg = None, None, None

        # layer 1: Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states_self, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            # use_cache=use_cache,
        )
        hidden_states_self = nn.functional.dropout(hidden_states_self, p=self.dropout, training=self.training)
        hidden_states_self = residual + hidden_states_self
        hidden_states_self = self.self_attn_layer_norm(hidden_states_self)

        # clamp inf values to enable fp16 training: from t5
        # if hidden_states_self.dtype in [torch.float16, torch.float32]:
        #     clamp_value = torch.where(
        #         torch.isinf(hidden_states_self).any(),
        #         torch.finfo(hidden_states_self.dtype).max - 1000,
        #         torch.finfo(hidden_states_self.dtype).max,
        #     )
        #     hidden_states_self = torch.clamp(hidden_states_self, min=-clamp_value, max=clamp_value)

        # layer 2: Pos Cross-Attention Block
        cross_attn_present_key_value_pos = None
        cross_attn_weights_pos = None
        if encoder_hidden_states_pos is not None:
            residual = hidden_states_self

            # pos cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            # cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states_pos, cross_attn_weights_pos, cross_attn_present_key_value_pos = self.encoder_attn_pos(
                hidden_states=hidden_states_self,
                key_value_states=encoder_hidden_states_pos,
                attention_mask=encoder_attention_mask_pos,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value_pos,
                output_attentions=output_attentions,
                # use_cache=use_cache,
            )
            hidden_states_pos = nn.functional.dropout(hidden_states_pos, p=self.dropout, training=self.training)
            hidden_states_pos = residual + hidden_states_pos
            hidden_states_pos = self.encoder_attn_layer_norm_pos(hidden_states_pos)

            # clamp inf values to enable fp16 training
            # if hidden_states_pos.dtype in [torch.float16, torch.float32]:
            #     clamp_value = torch.where(
            #         torch.isinf(hidden_states_pos).any(),
            #         torch.finfo(hidden_states_pos.dtype).max - 1000,
            #         torch.finfo(hidden_states_pos.dtype).max,
            #     )
            #     hidden_states_pos = torch.clamp(hidden_states_pos, min=-clamp_value, max=clamp_value)

        # layer 2.2: Neg Cross-Attention Block
        cross_attn_present_key_value_neg = None
        cross_attn_weights_neg = None
        if encoder_hidden_states_neg is not None:
            residual = hidden_states_self

            # neg cross_attn cached key/values tuple is at positions 5,6 of present_key_value tuple
            hidden_states_neg, cross_attn_weights_neg, cross_attn_present_key_value_neg = self.encoder_attn_neg(
                hidden_states=hidden_states_self,
                key_value_states=encoder_hidden_states_neg,
                attention_mask=encoder_attention_mask_neg,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value_neg,
                output_attentions=output_attentions,
                # use_cache=use_cache,
            )
            hidden_states_neg = nn.functional.dropout(hidden_states_neg, p=self.dropout, training=self.training)
            hidden_states_neg = residual + hidden_states_neg
            hidden_states_neg = self.encoder_attn_layer_norm_neg(hidden_states_neg)

            # clamp inf values to enable fp16 training
            # if hidden_states_neg.dtype in [torch.float16, torch.float32]:
            #     clamp_value = torch.where(
            #         torch.isinf(hidden_states_neg).any(),
            #         torch.finfo(hidden_states_neg.dtype).max - 1000,
            #         torch.finfo(hidden_states_neg.dtype).max,
            #     )
            #     hidden_states_neg = torch.clamp(hidden_states_neg, min=-clamp_value, max=clamp_value)

        # Combine self attn and cross attn key value states
        if present_key_value is not None: # tuple of (key_states, value_states) * 3
            present_key_value = present_key_value + cross_attn_present_key_value_pos + cross_attn_present_key_value_neg

        # combine two cross attention hidden states
        if self.args.model_variant == 'var6':
            hidden_states = self.alpha_self * hidden_states_self + self.alpha_pos * hidden_states_pos - self.alpha_neg * hidden_states_neg

        # tuple of three hidden states
            hidden_states_archive = (hidden_states_self, hidden_states_pos, hidden_states_neg) 
        # tuple of attention_outputs
            attention_outputs = (self_attn_weights, cross_attn_weights_pos, cross_attn_weights_neg)
        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        # clamp inf values to enable fp16 training
        # if hidden_states.dtype in [torch.float16, torch.float32]:
        #     clamp_value = torch.where(
        #         torch.isinf(hidden_states).any(),
        #         torch.finfo(hidden_states.dtype).max - 1000,
        #         torch.finfo(hidden_states.dtype).max,
        #     )
        #     hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += attention_outputs # plus 3 items into tuple

        if use_cache:
            outputs += (present_key_value,) # plus a tuple of 6

        outputs += (hidden_states_archive,) # plus a tuple of 3

        return outputs
    
class MyBartDecoder(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`MyBartDecoderBlock`]

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, args, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([MyBartDecoderBlock(config, args) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None, # decoder input ids
        attention_mask: Optional[torch.Tensor] = None, # decoder attention mask
        encoder_hidden_states_pos=None,
        encoder_attention_mask_pos=None,
        encoder_hidden_states_neg=None,
        encoder_attention_mask_neg=None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        """
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        """
        # boolean setting
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        ###################### prepare input ######################
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        
        ############ special for BART model: causal mask preparation############
        # past_key_values_length: decoder (target) sequence length, for self attention
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input) * self.embed_scale

        # print('deocder:::')
        # print("input shape:", input_shape, flush=True)
        # print("input embed shape:", inputs_embeds.shape)
        # print("attention_mask:", attention_mask.shape)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        # print("attention mask:", attention_mask)

        # expand encoder attention mask
        if encoder_hidden_states_pos is not None and encoder_attention_mask_pos is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask_pos = _expand_mask(encoder_attention_mask_pos, inputs_embeds.dtype, tgt_len=input_shape[-1])
        if encoder_hidden_states_neg is not None and encoder_attention_mask_neg is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask_neg = _expand_mask(encoder_attention_mask_neg, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input, past_key_values_length)
        positions = positions.to(inputs_embeds.device)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # print("hidden states shape:", hidden_states.shape, '\n')
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states_pos is not None and encoder_hidden_states_neg is not None) else None
        next_decoder_cache = () if use_cache else None # key_value_states
        all_hidden_states_archive  = ()

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )
        ###################### forward pass ######################
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states_pos,
                    encoder_attention_mask_pos,
                    encoder_hidden_states_neg,
                    encoder_attention_mask_neg,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states_pos=encoder_hidden_states_pos,
                    encoder_attention_mask_pos=encoder_attention_mask_pos,
                    encoder_hidden_states_neg=encoder_hidden_states_neg,
                    encoder_attention_mask_neg=encoder_attention_mask_neg,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            ###################### deal with output ######################
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[4 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states_pos is not None and encoder_hidden_states_neg is not None:
                    all_cross_attentions += ((layer_outputs[2],layer_outputs[3]),)
            all_hidden_states_archive += (layer_outputs[-1],)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, 
                          next_cache, 
                          all_hidden_states, 
                          all_self_attns, 
                          all_cross_attentions,
                          all_hidden_states_archive,
                          ]
                if v is not None
            )
        return MyBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            hidden_states_archive=all_hidden_states_archive,
        )

class MyBartEncoderLayer(nn.Module):
    """
    simply add clamp to hidden states
    """
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # new: clamp inf values to enable fp16 training
        if hidden_states.dtype in [torch.float16, torch.float32]:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
