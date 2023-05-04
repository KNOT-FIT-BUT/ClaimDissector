# Written by Martin Fajcik <martin.fajcik@vut.cz>
#
import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel, ElectraModel, RobertaModel, AutoConfig, BertModel
from transformers.activations import get_activation
from transformers.models.electra.modeling_electra import ElectraClassificationHead

try:
    from fast_transformers.builders import TransformerEncoderBuilder
except ImportError:
    pass


class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.out_proj = nn.Linear(config["hidden_size"], config["num_labels"])

    def forward(self, x):
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.training_steps = 0

        output_classes = 3
        cfg = AutoConfig.from_pretrained(config['verifier_transformer_type'], cache_dir=config["transformers_cache"])
        cfg.gradient_checkpointing = config.get("gradient_checkpointing", False)
        cfg.attention_probs_dropout_prob = config["hidden_dropout"]
        cfg.hidden_dropout_prob = config["hidden_dropout"]
        self.transformer = AutoModel.from_pretrained(config['verifier_transformer_type'], config=cfg,
                                                     cache_dir=config["transformers_cache"])
        if type(self.transformer) in [BertModel, RobertaModel]:
            # not used, so DDP complains
            del self.transformer.pooler
            self.transformer.pooler = None
        if type(self.transformer) == ElectraModel:
            self.classifier = ElectraClassificationHead({
                "hidden_size": self.transformer.config.hidden_size,
                "hidden_dropout_prob": self.transformer.config.hidden_dropout_prob,
                "num_labels": 1,
            })
        else:
            self.dropout = nn.Dropout(config["hidden_dropout"])
            if self.config.get("mh_head", False) or self.config.get("mh_w_sent_tokens", False):
                d_model = self.transformer.config.hidden_size
                multihead, mlp = [], []
                for i in range(self.config.get("mh_sent_layers", 1)):
                    multihead.append(torch.nn.MultiheadAttention(d_model, num_heads=4,
                                                                 batch_first=True,
                                                                 dropout=config["hidden_dropout"]))
                    d_inter = (d_model * 2) if self.config.get("mh_sent_residual", False) else d_model
                    if self.config.get("mh_sent_residual_lastlayer", None) is not None and \
                            i == (self.config.get("mh_sent_layers", 1) - 1):
                        d_inter = (d_model * 2) if self.config["mh_sent_residual_lastlayer"] else d_model

                    mlp.append(torch.nn.Sequential(torch.nn.LayerNorm(d_inter),
                                                   torch.nn.Linear(d_inter, d_model),
                                                   torch.nn.Dropout(config["hidden_dropout"]),
                                                   torch.nn.GELU()))
                self.multihead = nn.ModuleList(multihead)
                self.mlp = nn.ModuleList(mlp)
            if self.config.get("mh_clustered", False):
                self.transformer_clustered = TransformerEncoderBuilder.from_kwargs(**self.config['mh_clustered']).get()
            self.marg_classifier = nn.Linear(self.transformer.config.hidden_size, output_classes)

        self.init_weights(type(self.transformer))

    def init_weights(self, clz):
        """ Applies model's weight initialization to all non-pretrained parameters of this model"""
        for ch in self.children():
            if issubclass(ch.__class__, nn.Module) and not issubclass(ch.__class__, PreTrainedModel):
                ch.apply(lambda module: clz._init_weights(self.transformer, module))

    def set_gradient_checkpointing(self, enable):
        if enable:
            self.transformer.gradient_checkpointing_enable()
        else:
            self.transformer.gradient_checkpointing_disable()

    def forward(self, input_ids, token_type_ids, attention_mask, sentence_tokens_mask=None, inputs_embeds=None):
        outputs = self.transformer(input_ids, token_type_ids=token_type_ids,
                                   attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        pooled_output = outputs[0]
        pooled_output = self.dropout(pooled_output)

        if self.config.get("mh_head", False):
            all_passage_token_representations = pooled_output.view(-1, pooled_output.shape[-1]).unsqueeze(0)
            # TODO: keep it none now, try adding it later
            lin_attn_mask = None  # 1 - attention_mask.view(-1).unsqueeze(0)
            mh_output = \
                self.multihead(all_passage_token_representations, all_passage_token_representations,
                               all_passage_token_representations, key_padding_mask=lin_attn_mask)[
                    0]
            fused_output = torch.cat((pooled_output, mh_output.view(pooled_output.shape)), -1)
            pooled_output = self.mlp(fused_output)
        elif self.config.get("mh_clustered", False):
            # get rid of padding representations
            passage_mask = attention_mask.bool().view(-1)
            lin_pooled_output = pooled_output.view(-1, pooled_output.shape[-1])
            all_passage_token_representations = lin_pooled_output[passage_mask].unsqueeze(0)

            # do clustered MH (without mixed precision)
            # with torch.autocast(enabled=False, device_type='cuda'): # FOR HIGHER TORCH VERSION
            with torch.cuda.amp.autocast(enabled=False):
                pooled_output = self.transformer_clustered(all_passage_token_representations.float())
        elif self.config.get("mh_w_sent_tokens", False):
            # get rid of padding representations
            passage_mask = attention_mask.bool().view(-1)
            orig_shape = pooled_output.shape
            lin_pooled_output = pooled_output.view(-1, pooled_output.shape[-1])
            all_passage_token_representations = lin_pooled_output[passage_mask].unsqueeze(0)
            assert all_passage_token_representations.shape[1] == sentence_tokens_mask.shape[0]

            sentence_tokens_output = all_passage_token_representations[:, sentence_tokens_mask]

            mh_inputs = {
                "query": all_passage_token_representations,
                "key": sentence_tokens_output,
                "value": sentence_tokens_output
            }
            if self.config.get("dump_attention_matrices", False):
                mh_inputs.update({
                    "need_weights": self.config.get("dump_attention_matrices", False),
                })
            mh_attentions = []
            for i in range(len(self.multihead)):
                mh_output, mh_weights = \
                    self.multihead[i](**mh_inputs,
                                      key_padding_mask=None)
                mh_attentions.append(mh_weights)
                if (self.config.get("mh_sent_residual_lastlayer", None) is not None and \
                        i == (self.config.get("mh_sent_layers", 1) - 1)):
                    if self.config["mh_sent_residual_lastlayer"]:
                        mh_output = torch.cat((mh_inputs['query'], mh_output), -1)
                elif self.config.get("mh_sent_residual", False):
                    mh_output = torch.cat((mh_inputs['query'], mh_output), -1)

                new_representations = self.mlp[i](mh_output)

                mh_inputs = {
                    "query": new_representations,
                    "key": new_representations[:, sentence_tokens_mask],
                    "value": new_representations[:, sentence_tokens_mask]
                }
                if self.config.get("dump_attention_matrices", False):
                    mh_inputs.update({
                        "need_weights": self.config.get("dump_attention_matrices", False),
                        "average_attn_weights": False
                    })
            lin_pooled_output[passage_mask] = new_representations.squeeze(0).type(lin_pooled_output.dtype)

            pooled_output = lin_pooled_output.view(*orig_shape)
        marg_logits = self.marg_classifier(pooled_output)
        if self.config.get("dump_attention_matrices", False):
            return marg_logits, mh_attentions
        return marg_logits
