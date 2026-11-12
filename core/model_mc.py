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

class TopPPIMultiClass(nn.Module):

    def __init__(self, num_classes=7):
        super().__init__()

        print(f"Loading ESM-2 backbone for Multi-Class: {config.model_name}...")
        self.backbone = EsmModel.from_pretrained(config.model_name)

        if config.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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

        self.gat_adapter_v = StructureAwareGATAdapter(
            in_dim=config.embedding_dim,
            hidden_dim=config.gat_hidden_dim,
            heads=config.gat_heads,
            dropout=config.gat_dropout
        )
        self.gat_adapter_h = StructureAwareGATAdapter(
            in_dim=config.embedding_dim,
            hidden_dim=config.gat_hidden_dim,
            heads=config.gat_heads,
            dropout=config.gat_dropout
        )

        self.cross_attn_v2h = CrossAttentionLayer(
            config.gat_hidden_dim,
            config.cross_attn_heads,
            config.cross_attn_dropout
        )
        self.cross_attn_h2v = CrossAttentionLayer(
            config.gat_hidden_dim,
            config.cross_attn_heads,
            config.cross_attn_dropout
        )

        self.pool_v = AttentionPooling(config.gat_hidden_dim)
        self.pool_h = AttentionPooling(config.gat_hidden_dim)

        fused_dim = config.gat_hidden_dim * 4

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.LayerNorm(fused_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fused_dim // 2, num_classes)
        )

        self.projector = nn.Sequential(
            nn.Linear(fused_dim, config.projection_dim),
            nn.ReLU(),
            nn.Linear(config.projection_dim, config.projection_dim)
        )

        self.interaction_proj_v = nn.Linear(config.gat_hidden_dim, 1)
        self.interaction_proj_h = nn.Linear(config.gat_hidden_dim, 1)

    def forward(self, inputs_v, inputs_h):
        out_v = self.backbone(**inputs_v, output_attentions=True)
        out_h = self.backbone(**inputs_h, output_attentions=True)

        seq_emb_v = out_v.last_hidden_state
        seq_emb_h = out_h.last_hidden_state
        v_mask = inputs_v['attention_mask']
        h_mask = inputs_h['attention_mask']

        edge_index_v = self.graph_extractor(out_v.attentions, seq_emb_v, v_mask)
        edge_index_h = self.graph_extractor(out_h.attentions, seq_emb_h, h_mask)

        v_feat, _ = self.gat_adapter_v(seq_emb_v, edge_index_v, v_mask)
        h_feat, _ = self.gat_adapter_h(seq_emb_h, edge_index_h, h_mask)

        v_mask_bool = (v_mask == 0)
        h_mask_bool = (h_mask == 0)

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
        v_scores = torch.softmax(interaction_map.max(dim=2)[0], dim=1)
        h_scores = torch.softmax(interaction_map.max(dim=1)[0], dim=1)

        return logits, proj_features, (v_scores, h_scores)
StructPEFTCL_MC = TopPPIMultiClass
