import logging

import oneflow as flow
from oneflow import nn

from libai.layers import Linear
from libai.models.bert_model import BertModel
from libai.models.utils import init_method_normal

logger = logging.getLogger("libai." + __name__)


class SquadLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fct = nn.CrossEntropyLoss()
    
    def forward(self, start_logits, end_logits, start_position, end_position):
        start_loss = self.loss_fct(start_logits, start_position)
        end_loss = self.loss_fct(end_logits, end_position)
        return (start_loss + end_loss) / 2

class ModelForSquad(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = BertModel(cfg)
        if cfg.pretrain_megatron_weight is not None:
            from .load_megatron_weight import load_megatron_bert

            logger.info(f"loading pretraining: {cfg.pretrain_megatron_weight}")
            load_megatron_bert(self.model, cfg.pretrain_megatron_weight)
            logger.info("load succeed")

        init_method = init_method_normal(cfg.initializer_range)

        self.head = Linear(
            cfg.hidden_size,
            2,
            bias=True,
            parallel="row",
            init_method=init_method,
            layer_idx=-1,
        )
        self.loss_fct = SquadLoss()
    
    def forward(
        self,
        input_ids,  # [B, L]
        attention_mask,  # [B, L, L]
        start_position=None,  # [B]
        end_position=None,  # [B]
    ):
        encoder_output, _ = self.model(input_ids, attention_mask, token_type_ids=None)  # [B, L, H]
        logits = self.head(encoder_output)  # [B, L, 2]
        start_logits, end_logits = logits.split(1, dim=-1)  # [B, L, 1]
        start_logits = start_logits.squeeze(-1)  # [B, L]
        end_logits = end_logits.squeeze(-1)  # [B, L]

        if self.training and start_position is not None and end_position is not None:
            loss = self.loss_fct(start_logits, end_logits, start_position, end_position)
            return {"loss": loss}
        
        return {"start_logits": start_logits, "end_logits": end_logits}
