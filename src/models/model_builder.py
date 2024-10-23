import numpy as np
import inspect
from typing import List, Optional, Tuple, Union, Dict, Any
import warnings
import copy
import logging
import sys
sys.path.append('..')
sys.path.append('.')
import os
from os import path
import torch
from torch import nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from transformers import (
    T5ForConditionalGeneration, 
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    StoppingCriteriaList
)
from transformers.generation.stopping_criteria import validate_stopping_criteria
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer
from transformers.models.t5.modeling_t5 import T5Stack, T5PreTrainedModel, T5EncoderModel, T5ForConditionalGeneration
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.bart.modeling_bart import (
    BartConfig,
    BartEncoder,
    BartPretrainedModel,
    BartForConditionalGeneration,
    shift_tokens_right,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    ModelOutput,
    Seq2SeqModelOutput,
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions
)
# from transformers.utils import (
#     GreedySearchOutput,
#     GreedySearchEncoderDecoderOutput,
#     GreedySearchDecoderOnlyOutput
# )
import torch.nn.functional as F
from .module import (
    MyT5Decoder, 
    MySeq2SeqLMOutput, 
    MyGreedySearchEncoderDecoderOutput,
    MyBartDecoder,
    MyBeamSearchEncoderDecoderOutput,
)

import logging
logger = logging.getLogger(__name__)

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

def seq_len_to_mask(seq_len, max_len=None):
    """
    from fastNLP/core/utils.py
    change 1-d seq_leg to 2-d mask
    """
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask

def padding_seq_to_same_length(input_ids, max_pad_length, pad_token = 0):
    padding_length = max_pad_length - len(input_ids)
    padding_ids = [pad_token] * padding_length
    attention_mask = []

    # if padding_length <= 0, then truncate the input_ids
    if padding_length <= 0:
        attention_mask = [1] * max_pad_length
        input_ids = input_ids[:max_pad_length]
    # else pad the input_ids
    else:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + padding_ids
            
    assert len(input_ids) == max_pad_length
    assert len(attention_mask) == max_pad_length
  
    return input_ids, attention_mask

class MyBartModel(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"lm_head.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, model_type, args, config: BartConfig):
        super().__init__(config)
        self.config = config
        self.model_type = model_type
        self.args = args
        self.pad_id = config.pad_token_id
        self.hidden_size = config.d_model
        self.vocab_size = config.vocab_size

        self.shared = nn.Embedding(self.vocab_size, config.d_model, self.pad_id)
        self.encoder = BartEncoder(config, self.shared)
        self.decoder = MyBartDecoder(config, self.args, self.shared)

        self.register_buffer("final_logits_bias", torch.zeros((1, self.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.shared.num_embeddings, bias=False)

        # customized parameters
        self.margin = torch.tensor(1.0)
        if self.args.twolinearlayers:
            self.linear_layer_encoder = nn.Linear(self.hidden_size, self.hidden_size)
            self.linear_layer_decoder = nn.Linear(self.hidden_size, self.hidden_size)
        else:
            self.linear_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.alpha = torch.tensor(args.alpha) if args.alpha > 0.0 else nn.Parameter(torch.tensor(1.0))
        self.beta = torch.tensor(args.beta) if args.beta > 0.0 else nn.Parameter(torch.tensor(1.0))
        self.gamma = torch.tensor(args.gamma) if args.gamma > 0.0 else nn.Parameter(torch.tensor(1.0))

        # Initialize weights and apply final processing
        # self.post_init()
        self.init_weights()

        # load module_state_dict from pretrained model
        if args.mode == 'train':
            self.config.use_cache = False
            # TODO: check load model type
            _pretrained_model = BartForConditionalGeneration.from_pretrained(self.model_type).to(args.device)
            pretrained_model = _pretrained_model.model
            self.shared.load_state_dict(pretrained_model.shared.state_dict())
            self.encoder.load_state_dict(pretrained_model.encoder.state_dict())
            # initialize both cross_attention_weights in each decoder block with pretrained model cross_attention weights
            self.decoder.embed_tokens.load_state_dict(pretrained_model.decoder.embed_tokens.state_dict())
            self.decoder.embed_positions.load_state_dict(pretrained_model.decoder.embed_positions.state_dict())
            for b in range(config.decoder_layers):
                self.decoder.layers[b].self_attn.load_state_dict(pretrained_model.decoder.layers[b].self_attn.state_dict()) # self attention
                self.decoder.layers[b].activation_fn.load_state_dict(pretrained_model.decoder.layers[b].activation_fn.state_dict()) 
                self.decoder.layers[b].self_attn_layer_norm.load_state_dict(pretrained_model.decoder.layers[b].self_attn_layer_norm.state_dict())
                self.decoder.layers[b].encoder_attn_pos.load_state_dict(pretrained_model.decoder.layers[b].encoder_attn.state_dict()) # cross attention for pos
                self.decoder.layers[b].encoder_attn_layer_norm_pos.load_state_dict(pretrained_model.decoder.layers[b].encoder_attn_layer_norm.state_dict())
                self.decoder.layers[b].encoder_attn_neg.load_state_dict(pretrained_model.decoder.layers[b].encoder_attn.state_dict()) # cross attention for neg
                self.decoder.layers[b].encoder_attn_layer_norm_neg.load_state_dict(pretrained_model.decoder.layers[b].encoder_attn_layer_norm.state_dict())
                self.decoder.layers[b].fc1.load_state_dict(pretrained_model.decoder.layers[b].fc1.state_dict())
                self.decoder.layers[b].fc2.load_state_dict(pretrained_model.decoder.layers[b].fc2.state_dict())
                self.decoder.layers[b].final_layer_norm.load_state_dict(pretrained_model.decoder.layers[b].final_layer_norm.state_dict())
            self.decoder.layernorm_embedding.load_state_dict(pretrained_model.decoder.layernorm_embedding.state_dict())
            self.lm_head.load_state_dict(_pretrained_model.lm_head.state_dict())
        elif args.mode == 'test':
            # load state_dict from checkpoint in training script
            pass

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.shared
    
    def get_output_embeddings(self):
        return self.lm_head

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def affine_transformation(self, input_features, padding_mask, axis=1):
        """
        For decoder hidden states
        Aim: * reduce sequence length to 1 (reduce dim)
             * and normalize over the sequence length
        """
        # get masks that ignore the last </s> token
        length = torch.sum(padding_mask, axis=1) - 1 # tensor of size (batch,) of length values
        padding_mask = seq_len_to_mask(length, max_len=padding_mask.shape[-1]) # (batch, seq) of 0 and 1

        # length normalization
        if self.args.twolinearlayers:
            trans_tmp = F.relu(self.linear_layer_decoder(input_features))  # get (batch, seq, h_size)
        else:
            trans_tmp = F.relu(self.linear_layer(input_features))  # get (batch, seq, h_size)
        trans_tmp = trans_tmp * padding_mask.unsqueeze(-1).float() # padding_mask broadcast to last dim (repeated along new dim)相当于每个word的h_size要不要被pad
        trans_emb = torch.sum(trans_tmp, axis=axis) # (batch, h_size) sum over seq!!!!!!!
        return trans_emb * (1 / length.unsqueeze(-1))  # (batch, h_size) length normalization
    
    def affine_transformation_encoder(self, input_features, doc_length, axis=1):
        """
        For encoder hidden states
        Aim: * reduce sequence length to 1 (reduce dim)
             * and normalize over the sequence length
        input_features: decompose concatenated pos docs for one example in a batch: size of (seq_legnth, h_size) - > (doc_num, seq_length', h_size)
                        decompose query and neg doc
        doc_length: a list contains each pos doc length (neg doc length), need to ignore the first sentence and last token 
        * pos_num to replace batch size
        """
        # get original sequence length
        orig_seq_length = input_features.size(0)
        # print('doc length before process:', doc_length)
        # rest positions should be padded
        if orig_seq_length > sum(doc_length):
            doc_length[-1] = doc_length[-1] - 1 # the last token must be EOS
            doc_length.append(orig_seq_length - sum(doc_length))          
        else: #truncated
            if doc_length[-1] > 1: # don't -1 if the last sentence is only one token -> EOS
                doc_length[-1] = doc_length[-1] - 1
                doc_length.append(1) #EOS
        # length normalization
        if self.args.twolinearlayers:
            trans_tmp = F.relu(self.linear_layer_encoder(input_features))  # get (seq_len, h_size)
        else:
            trans_tmp = F.relu(self.linear_layer(input_features))  # get (seq_len, h_size)
        # print('doc length after process:', doc_length)
        split_features = torch.split(trans_tmp, doc_length, dim=0) # split the tensor into doc_num tensors, moved to GPU

        # summing and normalizing by length
        split_features = [torch.sum(f, axis=0) / l for f, l in zip(split_features, doc_length)]
        # return None if there is only query in the input
        if len(split_features) == 2:
            return None
        split_features = torch.stack(split_features[1:-1], dim=0) # (doc_num-2, h_size)
        
        return split_features # pos / neg doc features
    
    # @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids_pos: Optional[torch.LongTensor] = None, # positive encoder inputs
        attention_mask_pos: Optional[torch.FloatTensor] = None,
        input_ids_neg: Optional[torch.LongTensor] = None, # negative encoder inputs
        attention_mask_neg: Optional[torch.FloatTensor] = None,
        neg_sim_score: Optional[torch.FloatTensor] = None, # for training only: used in ranking loss: size of (sample_size) ==(batch_size)
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs_pos: Optional[Tuple[Tuple[torch.Tensor]]] = None, # positive encoder hidden states
        encoder_outputs_neg: Optional[Tuple[Tuple[torch.Tensor]]] = None, # negative encoder hidden states
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds_pos: Optional[torch.FloatTensor] = None,
        inputs_embeds_neg: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pos_doc_length: Optional[List] = None, # positive doc length
        neg_doc_length: Optional[List] = None, # negative doc length
        current_epoch: Optional[int] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        ################ Encode if needed (training, first prediction pass) ##################
        #### encoder hidden states for positive inputs ####
        if encoder_outputs_pos is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs_pos = self.encoder(
                input_ids=input_ids_pos,
                attention_mask=attention_mask_pos,
                inputs_embeds=inputs_embeds_pos,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs_pos, BaseModelOutput):
            encoder_outputs_pos = BaseModelOutput(
                last_hidden_state=encoder_outputs_pos[0],
                hidden_states=encoder_outputs_pos[1] if len(encoder_outputs_pos) > 1 else None,
                attentions=encoder_outputs_pos[2] if len(encoder_outputs_pos) > 2 else None,
            )
        ## get positive encoder hidden states
        hidden_states_pos = encoder_outputs_pos[0] # batch x seq_length x h_size

        #### encoder hidden states for negative inputs ####
        if encoder_outputs_neg is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs_neg = self.encoder(
                input_ids=input_ids_neg,
                attention_mask=attention_mask_neg,
                inputs_embeds=inputs_embeds_neg,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs_neg, BaseModelOutput):
            encoder_outputs_neg = BaseModelOutput(
                last_hidden_state=encoder_outputs_neg[0],
                hidden_states=encoder_outputs_neg[1] if len(encoder_outputs_neg) > 1 else None,
                attentions=encoder_outputs_neg[2] if len(encoder_outputs_neg) > 2 else None,
            )
        ## get negative encoder hidden states
        hidden_states_neg = encoder_outputs_neg[0] # batch x seq_length x h_size
        
        #################### Decode ####################
        # prepare decoder input ids and masks
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                # print("label size:", labels.shape)
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                # print("decoder_input_ids size:", decoder_input_ids.shape)
                # get decoder_attention_mask
                decoder_attention_mask = ~(decoder_input_ids == self.config.pad_token_id)
                decoder_attention_mask[:, 0] = 1 # bart decoder input ids start with 2  </s>
                # print("decoder_attention_mask size:", decoder_attention_mask.shape)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states_pos=hidden_states_pos,
            encoder_attention_mask_pos=attention_mask_pos,
            encoder_hidden_states_neg=hidden_states_neg,
            encoder_attention_mask_neg=attention_mask_neg,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True, # must be True
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        lm_logits = self.lm_head(sequence_output) # size of (batch, seq_length, vocab_size)
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        loss = None
        # print("check labels:", labels is not None, flush=True)
        if labels is not None:
            ################################### encoding loss #######################################
            if self.args.encoder_loss:
                # get each pos doc encoder feature, and neg doc encoder feature 
                encoder_loss = torch.tensor(0.0).to(self.args.device)
                sample_count = 0
                for i in range(len(hidden_states_pos)): # batch
                    # (check data structure, first length is query, second length is at least 1 due to </s>)
                    # skip the sample without neg doc
                    # & with only one pos for encoder loss: 
                    # # the first item is query length
                    if 1 in neg_doc_length[i] and len(pos_doc_length[i]) < 3:
                        continue
                    encoder_feature_pos = self.affine_transformation_encoder(hidden_states_pos[i], pos_doc_length[i], axis=1) # (pos_doc_num, h_size)
                    # print('hidden states neg norm:', torch.norm(hidden_states_neg[i].squeeze()), flush=True)
                    if 1 in neg_doc_length[i]:
                        encoder_feature_neg = None
                    else:
                        encoder_feature_neg = self.affine_transformation_encoder(hidden_states_neg[i], neg_doc_length[i], axis=1) # (neg_doc_num=1, h_size)

                    #### Compute positive-positive similarities
                    pos_distance = []
                    num_positives = len(encoder_feature_pos)
                    # print("num_pos:",num_positives)
                    if num_positives > 1:
                        for m in range(num_positives):
                            for n in range(m + 1, num_positives):
                                sim = F.cosine_similarity(encoder_feature_pos[m].unsqueeze(0), encoder_feature_pos[n].unsqueeze(0), dim=1)
                                distance = 1 - sim
                                pos_distance.append(distance)
                        # Stack all positive-positive distance, stack along a new dim=0
                        pos_distance = torch.stack(pos_distance)

                        pos_loss = pos_distance.mean()
                    else:
                        pos_loss = torch.tensor(0.0).to(self.args.device)

                    #### Compute positive-negative similarities
                    if encoder_feature_neg is not None:                        
                        pos_neg_distance = [1 - F.cosine_similarity(pos_feature.unsqueeze(0), encoder_feature_neg, dim=1)
                                                for pos_feature in encoder_feature_pos]
                        pos_neg_distance = torch.stack(pos_neg_distance)

                        neg_loss = F.relu(self.margin - pos_neg_distance).mean()
                    else:
                        neg_loss = torch.tensor(0.0).to(self.args.device)
    
                    total_loss = pos_loss + neg_loss
                    encoder_loss += total_loss
                    sample_count += 1

                # Get mean of encoder loss
                if sample_count != 0:
                    encoder_loss = encoder_loss / sample_count
                # else: encoder_loss = torch.tensor(0.0)
    
            ################################### generation loss #######################################
            # loss1 default cross entropy on lm_logits
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss1 = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            
            ################################### decoder CL loss #######################################
            if self.args.decoder_loss:
                if 'loss3trip' in self.args.loss_type:
                    # loss3: contrastive learning on the last decoder block hidden_states_archive
                    all_hidden_states_archive = decoder_outputs[-1]
                    _, last_hidden_states_pos, last_hidden_states_neg = all_hidden_states_archive[-1]
                    pos_feature = self.affine_transformation(last_hidden_states_pos, decoder_attention_mask) # batch x h_size
                    neg_feature = self.affine_transformation(last_hidden_states_neg, decoder_attention_mask) # batch x h_size
                    # get hidden_state of the last decoder block: the last second hidden_state from the hidden_states
                    last_block_hidden_states = decoder_outputs[2][-2][0]
                    decoder_feature = self.affine_transformation(last_block_hidden_states, decoder_attention_mask) # batch x h_size
                    # get cosine distance
                    cos_distance_pos = 1- torch.cosine_similarity(decoder_feature.unsqueeze(1), pos_feature.unsqueeze(1), dim=-1) # batch x 1
                    cos_distance_neg = 1- torch.cosine_similarity(decoder_feature.unsqueeze(1), neg_feature.unsqueeze(1), dim=-1) # batch x 1
                    # triplet loss
                    margin = torch.tensor(1.0)
                    triplet_loss = F.relu(cos_distance_pos - cos_distance_neg + margin) #max(0, value)
                    decoder_loss = triplet_loss.mean()
                if 'loss3nce' in self.args.loss_type:
                    all_hidden_states_archive = decoder_outputs[-1]
                    _, last_hidden_states_pos, last_hidden_states_neg = all_hidden_states_archive[-1]
                    pos_feature = self.affine_transformation(last_hidden_states_pos, decoder_attention_mask) # batch x h_size
                    neg_feature = self.affine_transformation(last_hidden_states_neg, decoder_attention_mask) # batch x h_size
                    # in-batch negative features
                    in_batch_neg_features = neg_feature.unsqueeze(0).repeat(neg_feature.size(0), 1, 1) # batch x batch x h_size
                    last_block_hidden_states = decoder_outputs[2][-2][0]
                    decoder_feature = self.affine_transformation(last_block_hidden_states, decoder_attention_mask) # batch x h_size
                    # get cosine distance
                    target_cos_distance = torch.cosine_similarity(decoder_feature.unsqueeze(1), pos_feature.unsqueeze(1), dim=-1) # batch x 1
                    contrastive_cos_distance = torch.cosine_similarity(decoder_feature.unsqueeze(1), in_batch_neg_features, dim=-1) # batch x batch
                    # cross entropy nce loss #
                    cl_cos_distance = torch.cat((target_cos_distance, contrastive_cos_distance), dim=1) # batch x (batch+1)
                    nce_labels = torch.zeros(cl_cos_distance.size(0), dtype=torch.long, device=self.args.device) # (batch_size)： class indices: postives are the 0-th
                    nce_loss = loss_fct(cl_cos_distance, nce_labels)
                    decoder_loss = nce_loss
            ######### ablation study #########
            if self.args.encoder_loss and self.args.decoder_loss:
                loss1= self.alpha /(self.alpha + self.beta + self.gamma) * loss1
                encoder_loss = self.beta /(self.alpha + self.beta + self.gamma) * encoder_loss
                decoder_loss = self.gamma /(self.alpha + self.beta + self.gamma) * decoder_loss
                total_loss = loss1 + encoder_loss + decoder_loss
                # create a tuple loss to contain total_loss, loss 1, encoder_loss, and decoder_loss
                loss = (total_loss, loss1.item(), encoder_loss.item(), decoder_loss.item())
            elif self.args.encoder_loss:
                loss1 = self.alpha /(self.alpha + self.beta) * loss1
                encoder_loss = self.beta /(self.alpha + self.beta) * encoder_loss
                total_loss = loss1 + encoder_loss
                loss = (total_loss, loss1.item(), encoder_loss.item())
            elif self.args.decoder_loss:
                loss1 = self.alpha /(self.alpha + self.gamma) * loss1
                decoder_loss =  self.gamma /(self.alpha + self.gamma) * decoder_loss
                total_loss = loss1 + decoder_loss
                loss = (total_loss, loss1.item(), decoder_loss.item())
            
            # ############### for visualization ###############
            # if output_hidden_states:
            #     all_hidden_states_archive = decoder_outputs[-1]
            #     _, last_hidden_states_pos, last_hidden_states_neg = all_hidden_states_archive[-1]

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs_pos + encoder_outputs_neg
            return ((loss,) + output) if loss is not None else output

        # TODO: 看要不要把encoder的pos和neg放进tuple里
        # 没把hidden_states_archive放进来
        return MySeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions, # self attention weights of each layer
            cross_attentions=decoder_outputs.cross_attentions,# cross attention weights of each layer: pos + neg
            encoder_last_hidden_state_pos=encoder_outputs_pos.last_hidden_state,
            encoder_hidden_states_pos=encoder_outputs_pos.hidden_states,
            encoder_attentions_pos=encoder_outputs_pos.attentions,
            encoder_last_hidden_state_neg=encoder_outputs_neg.last_hidden_state,
            encoder_hidden_states_neg=encoder_outputs_neg.hidden_states,
            encoder_attentions_neg=encoder_outputs_neg.attentions,
            # last_hidden_state_archive=(last_hidden_states_pos, last_hidden_states_neg),
        )
    
    # 有改动！TODO：check function! 生成 encoder_output_pos 和 neg
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. Prepare pos and neg encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        pos_appendix = "_pos"
        neg_appendix = "_neg"
        encoder_kwargs_general = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix) and not (argument.endswith(pos_appendix) or argument.endswith(neg_appendix))
        }
        encoder_kwargs_pos = {
            # get rid of '_pos' for the argument name
            argument[:-len(pos_appendix)]: value
            for argument, value in model_kwargs.items()
            if argument.endswith(pos_appendix)
        }
        encoder_kwargs_neg = {
            # get rid of '_neg' for the argument name
            argument[:-len(neg_appendix)]: value
            for argument, value in model_kwargs.items()
            if argument.endswith(neg_appendix)
        }
        # combine encoder_kwargs_general and encoder_kwargs_pos
        encoder_kwargs_pos = {**encoder_kwargs_general, **encoder_kwargs_pos}
        encoder_kwargs_neg = {**encoder_kwargs_general, **encoder_kwargs_neg}

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs_pos["return_dict"] = True
        encoder_kwargs_neg["return_dict"] = True
        model_kwargs["encoder_outputs_pos"]: ModelOutput = encoder(**encoder_kwargs_pos)
        model_kwargs["encoder_outputs_neg"]: ModelOutput = encoder(**encoder_kwargs_neg)

        return model_kwargs
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask_pos=None,
        attention_mask_neg=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs_pos=None,
        encoder_outputs_neg=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        # if past_key_values is not None:
        #     decoder_input_ids = decoder_input_ids[:, -1:]
        """
        被不同的decoding strategy调用，input_ids指的是decoder的输入， 不太需要encoder的input信息，需要encoder output和mask的信息
        adapt inputs_ids to decoder_input_ids when past_key_values is not None
        """
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length: # 如果input_ids的长度大于past_length
                remove_prefix_length = past_length # 那么past_length是要截断的长度
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1 #否则只取input_ids的最后一个token id

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # prepare decoder attention mask
        # decoder_attention_mask = ~(decoder_input_ids == self.config.pad_token_id)
        # decoder_attention_mask[:,0] = 1 # decoder input ids start with </s>

        return {
            # "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "past_key_values": past_key_values,
            "encoder_outputs_pos": encoder_outputs_pos,
            "encoder_outputs_neg": encoder_outputs_neg,
            "attention_mask_pos": attention_mask_pos,
            "attention_mask_neg": attention_mask_neg,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        #streamer: Optional["BaseStreamer"] = None, # no streamer function
        **model_kwargs,
    ) -> Union[MyGreedySearchEncoderDecoderOutput, torch.LongTensor]:
        r"""
        Main changes:
        * get encoder_attentions and encoder_hidden_states for pos and neg separately,
        * for output use
        """ 

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        ############################# customized part start #############################
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions_pos = model_kwargs["encoder_outputs_pos"].get("attentions") if output_attentions else None
            encoder_hidden_states_pos = (
                model_kwargs["encoder_outputs_pos"].get("hidden_states") if output_hidden_states else None
            )
            encoder_attentions_neg = model_kwargs["encoder_outputs_neg"].get("attentions") if output_attentions else None
            encoder_hidden_states_neg = (
                model_kwargs["encoder_outputs_neg"].get("hidden_states") if output_hidden_states else None
            ) 
        ############################# customized part end #############################
        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # print('greedy search:')
            # print('model_inputs, decoder input ids', model_inputs['decoder_input_ids'])

            # print('encoder output pos:', model_kwargs.encoder_outputs_pos)
            # print('model kargs:', model_kwargs)


            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :] #(batch, 1, vocab_size)

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                ############################# potential customized part start #############################
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            
            # checked, not need to modify
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return MyGreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions_pos=encoder_attentions_pos,
                    encoder_hidden_states_pos=encoder_hidden_states_pos,
                    encoder_attentions_neg=encoder_attentions_neg,
                    encoder_hidden_states_neg=encoder_hidden_states_neg,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            # else:
            #     return GreedySearchDecoderOnlyOutput(
            #         sequences=input_ids,
            #         scores=scores,
            #         attentions=decoder_attentions,
            #         hidden_states=decoder_hidden_states,
            #     )
        else:
            return input_ids

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        和bart本身的实现一样，没有改动 
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 4 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        # tuple of (key_states, value_states, k, v, k, v) * num_layers
        """
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            ) # index_select(dim, indix)
        return reordered_past

    @staticmethod
    # repeat input_ids and model_kwargs with size num_beams per batch: for beam search
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]
        example: [[2],[1]] * 3 -> [[2],[2],[2],[1],[1],[1]]
        """

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        # commented out from original sinippet
        if is_encoder_decoder:
            # if model_kwargs.get("encoder_outputs") is None:
            #     raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            ####### customized part #######
            model_kwargs["encoder_outputs_pos"] = _expand_dict_for_generation(model_kwargs["encoder_outputs_pos"])
            model_kwargs["encoder_outputs_neg"] = _expand_dict_for_generation(model_kwargs["encoder_outputs_neg"])

        return input_ids, model_kwargs
    
    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer, # 比greedy_search多了一个beam_scorer
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[MyBeamSearchEncoderDecoderOutput, torch.LongTensor]:
        r"""
        Parameters:
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.BeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt bist du?']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        ##### special part for beam search start ######
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape ###### important！

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )
        ##### special part for beam search end ######

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        ) # 初始化部分： 给每一个batch sample的每一个beam sample都设置一个空的tuple
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        # if return_dict_in_generate and self.config.is_encoder_decoder:
        #     encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        #     encoder_hidden_states = (
        #         model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        #     )
        ############################# customized part start #############################
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions_pos = model_kwargs["encoder_outputs_pos"].get("attentions") if output_attentions else None
            encoder_hidden_states_pos = (
                model_kwargs["encoder_outputs_pos"].get("hidden_states") if output_hidden_states else None
            )
            encoder_attentions_neg = model_kwargs["encoder_outputs_neg"].get("attentions") if output_attentions else None
            encoder_hidden_states_neg = (
                model_kwargs["encoder_outputs_neg"].get("hidden_states") if output_hidden_states else None
            ) 
        ############################# customized part end #############################
        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :] #(batch*num_beams, vocab_size)

            ##### special for beam search start ######
            # hack: adjust tokens for Marian model (MT). For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len) # do nothing for our model
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores) # shape alignment
            ##### special for beam search end ######

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,) # 应该要改一下

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            #注意这里改了shape
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            ) # (values, indices) is returned
            # 这里next_tokens的蕴含了beam dice和beam中的token indice
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")# beam indice
            next_tokens = next_tokens % vocab_size # 各beam对应的token indice

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids, #（batch*num_beam， seq_len）
                next_token_scores,
                next_tokens, # 本轮选中的beam中的next token indice
                next_indices, # 本轮选中的beam indice
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,#要更新的整体的beam_indices
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"] #应该是 （batch*num_beam，）

            # 更新input_ids
            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

            if return_dict_in_generate and output_scores:
                #更新要输出的beam_indices
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return MyBeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions_pos=encoder_attentions_pos,
                    encoder_hidden_states_pos=encoder_hidden_states_pos,
                    encoder_attentions_neg=encoder_attentions_neg,
                    encoder_hidden_states_neg=encoder_hidden_states_neg,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )

        else:
            return sequence_outputs["sequences"]

class MyT5Model(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, model_type, args, config: T5Config):
        super().__init__(config)
        self.model_type = model_type
        self.args = args
        self.pad_id = config.pad_token_id
        self.hidden_size = config.d_model
        self.vocab_size = config.vocab_size

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # self.temperature = torch.tensor(0.2)
        self.margin = torch.tensor(1.0)
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MyT5Decoder(decoder_config, self.args, self.shared)

        # self.linear_layer = nn.Linear(self.hidden_size, self.hidden_size)
        # self.alpha = torch.tensor(args.alpha) if args.alpha is not None else nn.Parameter(torch.tensor(1.0))
        # self.beta = torch.tensor(args.beta) if args.beta is not None else nn.Parameter(torch.tensor(1.0))
        # self.gamma = torch.tensor(args.gamma) if args.gamma is not None else nn.Parameter(torch.tensor(1.0))
        # customized parameters
        self.margin = torch.tensor(1.0)
        if self.args.twolinearlayers:
            self.linear_layer_encoder = nn.Linear(self.hidden_size, self.hidden_size)
            self.linear_layer_decoder = nn.Linear(self.hidden_size, self.hidden_size)
        else:
            self.linear_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.alpha = torch.tensor(args.alpha) if args.alpha > 0.0 else nn.Parameter(torch.tensor(1.0))
        self.beta = torch.tensor(args.beta) if args.beta > 0.0 else nn.Parameter(torch.tensor(1.0))
        self.gamma = torch.tensor(args.gamma) if args.gamma > 0.0 else nn.Parameter(torch.tensor(1.0))

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        # self.post_init()
        self.init_weights()

        # load module_state_dict from pretrained model
        if args.mode == 'train':
            pretrained_model = T5ForConditionalGeneration.from_pretrained(self.model_type).to(args.device)
            self.shared.load_state_dict(pretrained_model.shared.state_dict())
            self.encoder.load_state_dict(pretrained_model.encoder.state_dict())
            # initialize both cross_attention_weights in each decoder block with pretrained model cross_attention weights
            self.decoder.embed_tokens.load_state_dict(pretrained_model.decoder.embed_tokens.state_dict())
            for b in range(config.num_decoder_layers):
                self.decoder.block[b].layer[0].load_state_dict(pretrained_model.decoder.block[b].layer[0].state_dict()) # self attention
                self.decoder.block[b].layer[1].load_state_dict(pretrained_model.decoder.block[b].layer[1].state_dict()) # cross attention pos
                self.decoder.block[b].layer[2].load_state_dict(pretrained_model.decoder.block[b].layer[1].state_dict()) # cross attention neg
                self.decoder.block[b].layer[3].load_state_dict(pretrained_model.decoder.block[b].layer[2].state_dict()) # FFN
            self.decoder.final_layer_norm.load_state_dict(pretrained_model.decoder.final_layer_norm.state_dict())
            self.decoder.dropout.load_state_dict(pretrained_model.decoder.dropout.state_dict())
            self.lm_head.load_state_dict(pretrained_model.lm_head.state_dict())
        elif args.mode == 'test':
            # load state_dict from checkpoint in training script
            pass

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def linear_weight_scheduler(self, current_epoch):
        # start with 2/3, 1/3
        current_epoch += 1
        return max(1 - current_epoch / 3, 0)
    
    def affine_transformation(self, input_features, padding_mask, axis=1):
        """
        For decoder hidden states
        Aim: * reduce sequence length to 1 (reduce dim)
             * and normalize over the sequence length
        """
        # get masks that ignore the last </s> token
        length = torch.sum(padding_mask, axis=1) - 1 # tensor of size (batch,) of length values
        padding_mask = seq_len_to_mask(length, max_len=padding_mask.shape[-1]) # (batch, seq) of 0 and 1

        # length normalization
        if self.args.twolinearlayers:
            trans_tmp = F.relu(self.linear_layer_decoder(input_features))  # get (batch, seq, h_size)
        else:
            trans_tmp = F.relu(self.linear_layer(input_features))  # get (batch, seq, h_size)
        trans_tmp = trans_tmp * padding_mask.unsqueeze(-1).float() # padding_mask broadcast to last dim (repeated along new dim)相当于每个word的h_size要不要被pad
        trans_emb = torch.sum(trans_tmp, axis=axis) # (batch, h_size) sum over seq!!!!!!!
        return trans_emb * (1 / length.unsqueeze(-1))  # (batch, h_size) length normalization
    
    def affine_transformation_encoder(self, input_features, doc_length, axis=1):
        """
        For encoder hidden states
        Aim: * reduce sequence length to 1 (reduce dim)
             * and normalize over the sequence length
        input_features: decompose concatenated pos docs for one example in a batch: size of (seq_legnth, h_size) - > (doc_num, seq_length', h_size)
                        decompose query and neg doc
        doc_length: a list contains each pos doc length (neg doc length), need to ignore the first sentence and last token 
        * pos_num to replace batch size
        """
        # get original sequence length
        orig_seq_length = input_features.size(0)

        # rest positions should be padded
        if orig_seq_length > sum(doc_length):
            doc_length[-1] = doc_length[-1] - 1 # the last token must be EOS
            doc_length.append(orig_seq_length - sum(doc_length))          
        else: #truncated
            if doc_length[-1] > 1: # don't -1 if the last sentence is only one token -> EOS
                doc_length[-1] = doc_length[-1] - 1
                doc_length.append(1) #EOS
        # length normalization
        if self.args.twolinearlayers:
            trans_tmp = F.relu(self.linear_layer_encoder(input_features))  # get (seq_len, h_size)
        else:
            trans_tmp = F.relu(self.linear_layer(input_features))  # get (seq_len, h_size)
        split_features = torch.split(trans_tmp, doc_length, dim=0) # split the tensor into doc_num tensors, moved to GPU

        # summing and normalizing by length
        split_features = [torch.sum(f, axis=0) / l for f, l in zip(split_features, doc_length)]
        # return None if there is only query in the input
        if len(split_features) == 2:
            return None
        split_features = torch.stack(split_features[1:-1], dim=0) # (doc_num-2, h_size)
        
        return split_features # pos / neg doc features

    def ranking_loss_cont(self, cos_distance, neg_sim_score):
        '''
        cos_distance: batch x (1 + batch)
        neg_sim_score: batch x (1 + batch)
        '''
        # equivalent to initializing TotalLoss to 0
        # here is to avoid that some special samples will not go into the following for loop
        margin = 0.001 # similarity score dependent
        ones = torch.ones(cos_distance.size(), device=cos_distance.device) # batch x sample_size
        loss_func = torch.nn.MarginRankingLoss(0.0)
        total_loss = loss_func(cos_distance, cos_distance, ones)

        # candidate loss: C^2_K pairs in total
        n = cos_distance.size(1) # sample size = batch + 1
        # rank difference
        for i in range(1, n):
            pos_score = cos_distance[:, :-i]
            neg_score = cos_distance[:, i:]
            same_mask = (torch.abs(neg_sim_score[:, :-i] - neg_sim_score[:, i:]) > margin).float()
            ones = torch.ones(pos_score.size(), device=cos_distance.device)
            loss_func = torch.nn.MarginRankingLoss(margin * i, reduction='none')  # batch x i
            marginal_loss = loss_func(pos_score, neg_score, ones)
            if same_mask.sum() > 0:
                total_loss += (marginal_loss * same_mask).sum() / same_mask.sum()
        return total_loss
    
    def ranking_loss_basic(self, cos_distance, neg_sim_score):
        '''
        cos_distance: batch x (1 + batch)
        neg_sim_score: batch x (1 + batch)
        '''
        # equivalent to initializing TotalLoss to 0
        # here is to avoid that some special samples will not go into the following for loop
        margin = 0.001 # similarity score dependent
        ones = torch.ones(cos_distance.size(), device=cos_distance.device) # batch x sample_size
        loss_func = torch.nn.MarginRankingLoss(0.0)
        total_loss = loss_func(cos_distance, cos_distance, ones)

        # candidate loss: N (batch) pairs in total
        n = cos_distance.size(1) # sample size = batch + 1
        # rank difference
        for i in range(1, n):
            pos_score = cos_distance[:, 0]
            neg_score = cos_distance[:, i]
            same_mask = (torch.abs(neg_sim_score[:, 0] - neg_sim_score[:, i]) > margin).float()
            ones = torch.ones(pos_score.size(), device=cos_distance.device)
            loss_func = torch.nn.MarginRankingLoss(margin * i, reduction='none')  # batch x i
            marginal_loss = loss_func(pos_score, neg_score, ones)
            if same_mask.sum() > 0:
                total_loss += (marginal_loss * same_mask).sum() / same_mask.sum()
        return total_loss

    # @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids_pos: Optional[torch.LongTensor] = None, # positive encoder inputs
        attention_mask_pos: Optional[torch.FloatTensor] = None,
        input_ids_neg: Optional[torch.LongTensor] = None, # negative encoder inputs
        attention_mask_neg: Optional[torch.FloatTensor] = None,
        neg_sim_score: Optional[torch.FloatTensor] = None, # for training only: used in ranking loss: size of (sample_size) ==(batch_size)
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs_pos: Optional[Tuple[Tuple[torch.Tensor]]] = None, # positive encoder hidden states
        encoder_outputs_neg: Optional[Tuple[Tuple[torch.Tensor]]] = None, # negative encoder hidden states
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds_pos: Optional[torch.FloatTensor] = None,
        inputs_embeds_neg: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pos_doc_length: Optional[List] = None, # positive doc length
        neg_doc_length: Optional[List] = None, # negative doc length
        current_epoch: Optional[int] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], MySeq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        ################ Encode if needed (training, first prediction pass) ##################
        #### encoder hidden states for positive inputs ####
        if encoder_outputs_pos is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs_pos = self.encoder(
                input_ids=input_ids_pos,
                attention_mask=attention_mask_pos,
                inputs_embeds=inputs_embeds_pos,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs_pos, BaseModelOutput):
            encoder_outputs_pos = BaseModelOutput(
                last_hidden_state=encoder_outputs_pos[0],
                hidden_states=encoder_outputs_pos[1] if len(encoder_outputs_pos) > 1 else None,
                attentions=encoder_outputs_pos[2] if len(encoder_outputs_pos) > 2 else None,
            )
        ## get positive encoder hidden states
        hidden_states_pos = encoder_outputs_pos[0] # batch x seq_length x h_size

        #### encoder hidden states for negative inputs ####
        if encoder_outputs_neg is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs_neg = self.encoder(
                input_ids=input_ids_neg,
                attention_mask=attention_mask_neg,
                inputs_embeds=inputs_embeds_neg,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs_neg, BaseModelOutput):
            encoder_outputs_neg = BaseModelOutput(
                last_hidden_state=encoder_outputs_neg[0],
                hidden_states=encoder_outputs_neg[1] if len(encoder_outputs_neg) > 1 else None,
                attentions=encoder_outputs_neg[2] if len(encoder_outputs_neg) > 2 else None,
            )
        ## get negative encoder hidden states
        hidden_states_neg = encoder_outputs_neg[0] # batch x seq_length x h_size

        #################### Decode ####################
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

            # get deocoder_attention_mask
            decoder_attention_mask = ~(decoder_input_ids == self.config.pad_token_id)
            decoder_attention_mask[:, 0] = 1 # decoder input ids start with <pad>

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states_pos = hidden_states_pos.to(self.decoder.first_device)
            hidden_states_neg = hidden_states_neg.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask_pos is not None:
                attention_mask_pos = attention_mask_pos.to(self.decoder.first_device)
            if attention_mask_neg is not None:
                attention_mask_neg = attention_mask_neg.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states_pos=hidden_states_pos, #TOCheck
            encoder_attention_mask_pos=attention_mask_pos, #TODO
            encoder_hidden_states_neg=hidden_states_neg, #TODO
            encoder_attention_mask_neg=attention_mask_neg, #TODO
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True, # must be True
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.hidden_size**-0.5)

        lm_logits = self.lm_head(sequence_output) # size of (batch, seq_length, vocab_size)

        loss = None
        if labels is not None:
            ################################### encoding loss #######################################
            if self.args.encoder_loss:
                ################################### encoding loss #######################################
                # get each pos doc encoder feature, and neg doc encoder feature 
                encoder_loss = torch.tensor(0.0).to(self.args.device)
                sample_count = 0
                for i in range(len(hidden_states_pos)): # batch
                    # hidden_states_pos[i] = hidden_states_pos[i].detach() TODO: detach or not

                    # skip the sample without neg doc & with only one pos for encoder loss
                    if len(neg_doc_length[i]) < 2 and len(pos_doc_length[i]) < 3:
                        continue
                    # pos_len = torch.tensor(pos_doc_length[i]).to(self.args.device)
                    # neg_len = torch.tensor(neg_doc_length[i]).to(self.args.device)
                    encoder_feature_pos = self.affine_transformation_encoder(hidden_states_pos[i], pos_doc_length[i], axis=1) # (pos_doc_num, h_size)
                    encoder_feature_neg = self.affine_transformation_encoder(hidden_states_neg[i], neg_doc_length[i], axis=1) # (neg_doc_num=1, h_size)

                    # print('encoder_feature_pos:', encoder_feature_pos.size())
                    # print('encoder_feature_neg:', encoder_feature_neg.size())

                    # Compute positive-positive similarities
                    pos_distance = []
                    num_positives = len(encoder_feature_pos)
                    # print("num_pos:",num_positives)
                    if num_positives > 1:
                        for i in range(num_positives):
                            for j in range(i + 1, num_positives):
                                sim = F.cosine_similarity(encoder_feature_pos[i].unsqueeze(0), encoder_feature_pos[j].unsqueeze(0), dim=1)
                                distance = 1 - sim
                                pos_distance.append(distance)
                        # Stack all positive-positive distance, stack along a new dim=0
                        pos_distance = torch.stack(pos_distance)

                        # Positive-positive loss (pull them closer)
                        ## deprecated: log
                        # pos_loss = -torch.log(torch.exp(pos_distance / self.temperature).mean())
                        pos_loss = pos_distance.mean()
                    else:
                        pos_loss = torch.tensor(0.0).to(self.args.device)

                    # Compute positive-negative similarities
                    if encoder_feature_neg is not None:                        
                        pos_neg_distance = [1 - F.cosine_similarity(pos_feature.unsqueeze(0), encoder_feature_neg, dim=1)
                                                for pos_feature in encoder_feature_pos]

                        # Stack all positive-negative similarities
                        pos_neg_distance = torch.stack(pos_neg_distance)
                        # print('pos_neg_sim:', pos_neg_distance)

                        # Positive-negative loss (push them apart)
                        ## deprecated: sim 
                        # neg_loss = F.relu(pos_neg_distance - self.margin).mean()
                        # instead: distance:
                        neg_loss = F.relu(self.margin - pos_neg_distance).mean()
                    else:
                        neg_loss = torch.tensor(0.0).to(self.args.device)
    
                    # Combine the losses
                    total_loss = pos_loss + neg_loss
                    # print('pos loss:', pos_loss)
                    # print('neg loss:', neg_loss)
                    encoder_loss += total_loss
                    sample_count += 1
                
                # Get mean of encoder loss
                encoder_loss = encoder_loss / sample_count
                # print('encoder_loss', encoder_loss)
                # raise NotImplementedError
    
            ################################### generation loss #######################################
            # loss1 default cross entropy on lm_logits
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss1 = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            ################################### decoder CL loss #######################################
            if self.args.decoder_loss:
                if 'loss3trip' in self.args.loss_type:
                    # loss3: contrastive learning on the last decoder block hidden_states_archive
                    all_hidden_states_archive = decoder_outputs[-1]
                    _, last_hidden_states_pos, last_hidden_states_neg = all_hidden_states_archive[-1]
                    pos_feature = self.affine_transformation(last_hidden_states_pos, decoder_attention_mask) # batch x h_size
                    neg_feature = self.affine_transformation(last_hidden_states_neg, decoder_attention_mask) # batch x h_size
                    # get hidden_state of the last decoder block: the last second hidden_state from the hidden_states
                    last_block_hidden_states = decoder_outputs[2][-2][0]
                    decoder_feature = self.affine_transformation(last_block_hidden_states, decoder_attention_mask) # batch x h_size
                    # get cosine distance
                    cos_distance_pos = 1- torch.cosine_similarity(decoder_feature.unsqueeze(1), pos_feature.unsqueeze(1), dim=-1) # batch x 1
                    cos_distance_neg = 1- torch.cosine_similarity(decoder_feature.unsqueeze(1), neg_feature.unsqueeze(1), dim=-1) # batch x 1
                    # triplet loss
                    margin = torch.tensor(1.0)
                    triplet_loss = F.relu(cos_distance_pos - cos_distance_neg + margin) #max(0, value)
                    decoder_loss = triplet_loss.mean()
                if 'loss3nce' in self.args.loss_type:
                    all_hidden_states_archive = decoder_outputs[-1]
                    _, last_hidden_states_pos, last_hidden_states_neg = all_hidden_states_archive[-1]
                    pos_feature = self.affine_transformation(last_hidden_states_pos, decoder_attention_mask) # batch x h_size
                    neg_feature = self.affine_transformation(last_hidden_states_neg, decoder_attention_mask) # batch x h_size
                    # in-batch negative features
                    in_batch_neg_features = neg_feature.unsqueeze(0).repeat(neg_feature.size(0), 1, 1) # batch x batch x h_size
                    last_block_hidden_states = decoder_outputs[2][-2][0]
                    decoder_feature = self.affine_transformation(last_block_hidden_states, decoder_attention_mask) # batch x h_size
                    # get cosine distance
                    target_cos_distance = torch.cosine_similarity(decoder_feature.unsqueeze(1), pos_feature.unsqueeze(1), dim=-1) # batch x 1
                    contrastive_cos_distance = torch.cosine_similarity(decoder_feature.unsqueeze(1), in_batch_neg_features, dim=-1) # batch x batch
                    # cross entropy nce loss #
                    cl_cos_distance = torch.cat((target_cos_distance, contrastive_cos_distance), dim=1) # batch x (batch+1)
                    nce_labels = torch.zeros(cl_cos_distance.size(0), dtype=torch.long, device=self.args.device) # (batch_size)： class indices: postives are the 0-th
                    nce_loss = loss_fct(cl_cos_distance, nce_labels)
                    decoder_loss = nce_loss

            ######### ablation study #########
            if self.args.encoder_loss and self.args.decoder_loss:
                loss1= self.alpha /(self.alpha + self.beta + self.gamma) * loss1
                encoder_loss = self.beta /(self.alpha + self.beta + self.gamma) * encoder_loss
                decoder_loss = self.gamma /(self.alpha + self.beta + self.gamma) * decoder_loss
                total_loss = loss1 + encoder_loss + decoder_loss
                # create a tuple loss to contain total_loss, loss 1, encoder_loss, and decoder_loss
                loss = (total_loss, loss1.item(), encoder_loss.item(), decoder_loss.item())
            elif self.args.encoder_loss:
                loss1 = self.alpha /(self.alpha + self.beta) * loss1
                encoder_loss = self.beta /(self.alpha + self.beta) * encoder_loss
                total_loss = loss1 + encoder_loss
                loss = (total_loss, loss1.item(), encoder_loss.item())
            elif self.args.decoder_loss:
                loss1 = self.alpha /(self.alpha + self.gamma) * loss1
                decoder_loss =  self.gamma /(self.alpha + self.gamma) * decoder_loss
                total_loss = loss1 + decoder_loss
                loss = (total_loss, loss1.item(), decoder_loss.item())

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs_pos + encoder_outputs_neg
            return ((loss,) + output) if loss is not None else output

        return MySeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions, # self attention weights of each layer
            cross_attentions=decoder_outputs.cross_attentions,# cross attention weights of each layer: pos + neg
            encoder_last_hidden_state_pos=encoder_outputs_pos.last_hidden_state,
            encoder_hidden_states_pos=encoder_outputs_pos.hidden_states,
            encoder_attentions_pos=encoder_outputs_pos.attentions,
            encoder_last_hidden_state_neg=encoder_outputs_neg.last_hidden_state,
            encoder_hidden_states_neg=encoder_outputs_neg.hidden_states,
            encoder_attentions_neg=encoder_outputs_neg.attentions,
        )
    
    # 有改动！TODO：check function! 生成 encoder_output_pos 和 neg
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. Prepare pos and neg encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        pos_appendix = "_pos"
        neg_appendix = "_neg"
        encoder_kwargs_general = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix) and not (argument.endswith(pos_appendix) or argument.endswith(neg_appendix))
        }
        encoder_kwargs_pos = {
            # get rid of '_pos' for the argument name
            argument[:-len(pos_appendix)]: value
            for argument, value in model_kwargs.items()
            if argument.endswith(pos_appendix)
        }
        encoder_kwargs_neg = {
            # get rid of '_neg' for the argument name
            argument[:-len(neg_appendix)]: value
            for argument, value in model_kwargs.items()
            if argument.endswith(neg_appendix)
        }
        # combine encoder_kwargs_general and encoder_kwargs_pos
        encoder_kwargs_pos = {**encoder_kwargs_general, **encoder_kwargs_pos}
        encoder_kwargs_neg = {**encoder_kwargs_general, **encoder_kwargs_neg}

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs_pos["return_dict"] = True
        encoder_kwargs_neg["return_dict"] = True
        model_kwargs["encoder_outputs_pos"]: ModelOutput = encoder(**encoder_kwargs_pos)
        model_kwargs["encoder_outputs_neg"]: ModelOutput = encoder(**encoder_kwargs_neg)

        return model_kwargs

    # 有改动： prepare decoder attention mask to replace _prepare_attention_mask_for_generation() in generate.utils()
    def prepare_inputs_for_generation(
        self,
        input_ids, # 是decoder的输入
        past_key_values=None,
        attention_mask_pos=None,
        attention_mask_neg=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs_pos=None,
        encoder_outputs_neg=None,
        **kwargs,
    ):
        """
        被不同的decoding strategy调用，input_ids指的是decoder的输入， 不太需要encoder的input信息，需要encoder output和mask的信息
        adapt inputs_ids to decoder_input_ids when past_key_values is not None
        """
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length: # 如果input_ids的长度大于past_length
                remove_prefix_length = past_length # 那么past_length是要截断的长度
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1 #否则只取input_ids的最后一个token id

            input_ids = input_ids[:, remove_prefix_length:]

        # prepare decoder attention mask
        decoder_attention_mask = ~(input_ids == self.config.pad_token_id)
        decoder_attention_mask[:,0] = 1 # decoder input ids start with <pad>

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs_pos": encoder_outputs_pos,
            "encoder_outputs_neg": encoder_outputs_neg,
            "attention_mask_pos": attention_mask_pos,
            "attention_mask_neg": attention_mask_neg,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        #streamer: Optional["BaseStreamer"] = None, # no streamer function
        **model_kwargs,
    ) -> Union[MyGreedySearchEncoderDecoderOutput, torch.LongTensor]:
        r"""
        Main changes:
        * get encoder_attentions and encoder_hidden_states for pos and neg separately,
        * for output use
        """ 

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        ############################# customized part start #############################
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions_pos = model_kwargs["encoder_outputs_pos"].get("attentions") if output_attentions else None
            encoder_hidden_states_pos = (
                model_kwargs["encoder_outputs_pos"].get("hidden_states") if output_hidden_states else None
            )
            encoder_attentions_neg = model_kwargs["encoder_outputs_neg"].get("attentions") if output_attentions else None
            encoder_hidden_states_neg = (
                model_kwargs["encoder_outputs_neg"].get("hidden_states") if output_hidden_states else None
            ) 
        ############################# customized part end #############################
        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :] #(batch, 1, vocab_size)

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                ############################# potential customized part start #############################
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            
            # checked, not need to modify
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return MyGreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions_pos=encoder_attentions_pos,
                    encoder_hidden_states_pos=encoder_hidden_states_pos,
                    encoder_attentions_neg=encoder_attentions_neg,
                    encoder_hidden_states_neg=encoder_hidden_states_neg,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            # else:
            #     return GreedySearchDecoderOnlyOutput(
            #         sequences=input_ids,
            #         scores=scores,
            #         attentions=decoder_attentions,
            #         hidden_states=decoder_hidden_states,
            #     )
        else:
            return input_ids
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        和bart本身的实现一样，没有改动 
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 4 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        # tuple of (key_states, value_states, k, v, k, v) * num_layers
        """
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            ) # index_select(dim, indix)
        return reordered_past

    @staticmethod
    # repeat input_ids and model_kwargs with size num_beams per batch: for beam search
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]
        example: [[2],[1]] * 3 -> [[2],[2],[2],[1],[1],[1]]
        """

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        # commented out from original sinippet
        if is_encoder_decoder:
            # if model_kwargs.get("encoder_outputs") is None:
            #     raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            ####### customized part #######
            model_kwargs["encoder_outputs_pos"] = _expand_dict_for_generation(model_kwargs["encoder_outputs_pos"])
            model_kwargs["encoder_outputs_neg"] = _expand_dict_for_generation(model_kwargs["encoder_outputs_neg"])

        return input_ids, model_kwargs
    
    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer, # 比greedy_search多了一个beam_scorer
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[MyBeamSearchEncoderDecoderOutput, torch.LongTensor]:
        r"""
        Parameters:
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.BeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt bist du?']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        ##### special part for beam search start ######
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape ###### important！

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )
        ##### special part for beam search end ######

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        ) # 初始化部分： 给每一个batch sample的每一个beam sample都设置一个空的tuple
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        # if return_dict_in_generate and self.config.is_encoder_decoder:
        #     encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        #     encoder_hidden_states = (
        #         model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        #     )
        ############################# customized part start #############################
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions_pos = model_kwargs["encoder_outputs_pos"].get("attentions") if output_attentions else None
            encoder_hidden_states_pos = (
                model_kwargs["encoder_outputs_pos"].get("hidden_states") if output_hidden_states else None
            )
            encoder_attentions_neg = model_kwargs["encoder_outputs_neg"].get("attentions") if output_attentions else None
            encoder_hidden_states_neg = (
                model_kwargs["encoder_outputs_neg"].get("hidden_states") if output_hidden_states else None
            ) 
        ############################# customized part end #############################
        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :] #(batch*num_beams, vocab_size)

            ##### special for beam search start ######
            # hack: adjust tokens for Marian model (MT). For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len) # do nothing for our model
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores) # shape alignment
            ##### special for beam search end ######

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,) # 应该要改一下

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            #注意这里改了shape
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            ) # (values, indices) is returned
            # 这里next_tokens的蕴含了beam dice和beam中的token indice
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")# beam indice
            next_tokens = next_tokens % vocab_size # 各beam对应的token indice

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids, #（batch*num_beam， seq_len）
                next_token_scores,
                next_tokens, # 本轮选中的beam中的next token indice
                next_indices, # 本轮选中的beam indice
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,#要更新的整体的beam_indices
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"] #应该是 （batch*num_beam，）

            # 更新input_ids
            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

            if return_dict_in_generate and output_scores:
                #更新要输出的beam_indices
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return MyBeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions_pos=encoder_attentions_pos,
                    encoder_hidden_states_pos=encoder_hidden_states_pos,
                    encoder_attentions_neg=encoder_attentions_neg,
                    encoder_hidden_states_neg=encoder_hidden_states_neg,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            # else:
            #     return BeamSearchDecoderOnlyOutput(
            #         sequences=sequence_outputs["sequences"],
            #         sequences_scores=sequence_outputs["sequence_scores"],
            #         scores=scores,
            #         beam_indices=sequence_outputs["beam_indices"],
            #         attentions=decoder_attentions,
            #         hidden_states=decoder_hidden_states,
                #  )
        else:
            return sequence_outputs["sequences"]

