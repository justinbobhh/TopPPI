import torch
import torch.nn as nn
from transformers import EsmModel
from peft import get_peft_model, LoraConfig, TaskType
from config import config
from layers import (
    ImplicitContactMapExtractor,
    StructureAwareGATAdapter,
    CrossAttentionLayer,
    AttentionPooling
)

class TopPPI(nn.Module):
    def __init__(self):
        super().__init__()

        print(f"Loading ESM-2 backbone: {config.model_name}...")
        self.backbone = EsmModel.from_pretrained(config.model_name)

        if config.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        for param in self.backbone.parameters():
            param.requires_grad = False

        if config.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules
            )
            self.backbone = get_peft_model(self.backbone, peft_config)

        self.graph_extractor = ImplicitContactMapExtractor(
            layer_idx=config.contact_head_layer,
            head_indices=config.contact_head_indices,
            threshold=config.contact_threshold,
            k_neighbors=config.knn_k
        )

        self.gat_adapter = StructureAwareGATAdapter(
            in_dim=config.embedding_dim,
            hidden_dim=config.gat_hidden_dim,
            heads=config.gat_heads,
            dropout=config.gat_dropout
        )

        self.cross_attn_v2h = CrossAttentionLayer(config.gat_hidden_dim, num_heads=config.cross_attn_heads)
        self.cross_attn_h2v = CrossAttentionLayer(config.gat_hidden_dim, num_heads=config.cross_attn_heads)

        self.pool_v = AttentionPooling(config.gat_hidden_dim)
        self.pool_h = AttentionPooling(config.gat_hidden_dim)

        fused_dim = config.gat_hidden_dim * 4

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.LayerNorm(fused_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fused_dim // 2, 1)
        )

        self.projector = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Linear(fused_dim // 2, config.projection_dim)
        )

        self.interaction_proj_v = nn.Linear(config.gat_hidden_dim, config.gat_hidden_dim)
        self.interaction_proj_h = nn.Linear(config.gat_hidden_dim, config.gat_hidden_dim)

    def forward_single_branch(self, input_ids, attention_mask):

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        sequence_output = outputs.last_hidden_state
        attentions = outputs.attentions

        edge_indices = self.graph_extractor(attentions, sequence_output, attention_mask)
        gat_feat, gat_mask = self.gat_adapter(sequence_output, edge_indices, attention_mask)

        return gat_feat, gat_mask

    def forward(self, virus_inputs, host_inputs):

        v_feat, v_mask = self.forward_single_branch(virus_inputs['input_ids'], virus_inputs['attention_mask'])
        h_feat, h_mask = self.forward_single_branch(host_inputs['input_ids'], host_inputs['attention_mask'])

        v_mask_bool = ~v_mask.bool()
        h_mask_bool = ~h_mask.bool()

        v_cross = self.cross_attn_v2h(query=v_feat, key=h_feat, value=h_feat, key_padding_mask=h_mask_bool)
        h_cross = self.cross_attn_h2v(query=h_feat, key=v_feat, value=v_feat, key_padding_mask=v_mask_bool)

        v_global, _ = self.pool_v(v_cross, mask=v_mask.bool())
        h_global, _ = self.pool_h(h_cross, mask=h_mask.bool())

        fused_mult = v_global * h_global
        fused_diff = torch.abs(v_global - h_global)
        fused_repr = torch.cat([v_global, h_global, fused_mult, fused_diff], dim=1)

        logits = self.classifier(fused_repr)
        proj_features = self.projector(fused_repr)

        v_proj = torch.tanh(self.interaction_proj_v(v_cross))
        h_proj = torch.tanh(self.interaction_proj_h(h_cross))

        interaction_map = torch.bmm(v_proj, h_proj.transpose(1, 2))

        combined_mask = torch.bmm(v_mask.unsqueeze(2).float(), h_mask.unsqueeze(1).float())
        interaction_map = interaction_map.masked_fill(combined_mask == 0, -1e4)

        v_scores = torch.softmax(interaction_map.mean(dim=2), dim=1)
        h_scores = torch.softmax(interaction_map.mean(dim=1), dim=1)

        return logits, proj_features, (v_scores, h_scores)

StructPEFTCL = TopPPI
