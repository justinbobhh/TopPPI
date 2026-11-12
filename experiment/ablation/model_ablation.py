import os as _os
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "core" / "config.py").is_file())
_sys.path.insert(0, str(_ROOT / "core"))
_sys.path.append(str(_ROOT / "experiment" / "ablation"))
_os.environ["TOPPPI_TASK"] = 'yeast'

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.utils import remove_self_loops
from transformers import EsmModel

from peft import LoraConfig, TaskType, get_peft_model

from config import config
from layers import CrossAttentionLayer, AttentionPooling, StructureAwareGATAdapter
from model import TopPPI

ABLATION_VARIANTS = {
    "full": {
        "use_lora": True,
        "graph_mode": "attention_knn",
        "use_cross_attn": True,
        "description": "Full TopPPI model.",
    },
    "wo_lora": {
        "use_lora": False,
        "graph_mode": "attention_knn",
        "use_cross_attn": True,
        "description": "Remove LoRA; ESM-2 stays frozen.",
    },
    "wo_struct_gat": {
        "use_lora": True,
        "graph_mode": "none",
        "use_cross_attn": True,
        "description": "Remove PLM-derived topology construction and the GAT adapter.",
    },
    "attention_only": {
        "use_lora": True,
        "graph_mode": "attention",
        "use_cross_attn": True,
        "description": "Use only ESM self-attention-derived graph edges, without KNN edges.",
    },
    "knn_only": {
        "use_lora": True,
        "graph_mode": "knn",
        "use_cross_attn": True,
        "description": "Use only KNN graph, without ESM attention contact edges.",
    },
    "wo_cross_attn": {
        "use_lora": True,
        "graph_mode": "attention_knn",
        "use_cross_attn": False,
        "description": "Remove bidirectional protein-pair cross-attention.",
    },
}

_VARIANT_ALIASES = {
    "no_lora": "wo_lora",
    "no_struct_gat": "wo_struct_gat",
    "without_struct_gat": "wo_struct_gat",
    "contact_only": "attention_only",
    "no_cross_attn": "wo_cross_attn",
    "without_cross_attn": "wo_cross_attn",
}

def normalize_variant(variant):
    variant = variant.lower()
    variant = _VARIANT_ALIASES.get(variant, variant)
    if variant not in ABLATION_VARIANTS:
        valid = ", ".join(available_variants())
        raise ValueError(f"Unknown ablation variant '{variant}'. Valid variants: {valid}")
    return variant

def available_variants():
    return list(ABLATION_VARIANTS.keys())

def resolve_model_name(model_name):

    raw_path = Path(model_name)
    if raw_path.exists():
        return str(raw_path)

    model_dir = Path(__file__).resolve().parent
    project_root = model_dir.parent
    candidates = [
        model_dir / model_name,
        project_root / model_name,
        project_root / Path(model_name).name,
        Path.cwd() / model_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return model_name

class AblationGraphExtractor(nn.Module):

    def __init__(self, layer_idx, head_indices, threshold=0.1, k_neighbors=10, graph_mode="attention_knn"):
        super().__init__()
        if graph_mode not in {"attention_knn", "attention", "knn"}:
            raise ValueError(f"Unsupported graph_mode: {graph_mode}")
        self.layer_idx = layer_idx
        self.head_indices = head_indices
        self.threshold = threshold
        self.k = k_neighbors
        self.graph_mode = graph_mode

    def forward(self, attention_outputs, sequence_embeddings, attention_mask):

        layer_attentions = attention_outputs[self.layer_idx]
        batch_size = layer_attentions.size(0)
        edge_indices_list = []

        for b in range(batch_size):
            valid_len = int(attention_mask[b].sum().item())
            device = sequence_embeddings.device
            edge_parts = []

            if valid_len <= 0:
                edge_indices_list.append(
                    torch.empty((2, 0), dtype=torch.long, device=device)
                )
                continue

            if self.graph_mode in {"attention_knn", "attention"}:
                selected_heads = layer_attentions[
                    b, self.head_indices, :valid_len, :valid_len
                ]
                contact_map = selected_heads.mean(dim=0)
                mask_contact = contact_map > self.threshold
                mask_contact.fill_diagonal_(0)
                edge_index_contact = mask_contact.nonzero().t()
                if edge_index_contact.numel() > 0:
                    edge_parts.append(edge_index_contact)

            if self.graph_mode in {"attention_knn", "knn"}:
                curr_emb = sequence_embeddings[b, :valid_len, :]
                dist_matrix = torch.cdist(curr_emb, curr_emb)
                k_val = min(self.k + 1, valid_len)
                _, indices = dist_matrix.topk(k=k_val, dim=1, largest=False)
                source_nodes = (
                    torch.arange(valid_len, device=device)
                    .unsqueeze(1)
                    .expand(-1, k_val)
                )
                edge_index_knn = torch.stack(
                    [source_nodes.flatten(), indices.flatten()], dim=0
                )
                if edge_index_knn.numel() > 0:
                    edge_parts.append(edge_index_knn)

            if edge_parts:
                edge_index = torch.cat(edge_parts, dim=1)
                edge_index, _ = remove_self_loops(edge_index)
                edge_index = torch.unique(edge_index, dim=1)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

            edge_indices_list.append(edge_index)

        return edge_indices_list

class SequenceProjectionAdapter(nn.Module):

    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        h = self.proj(x)
        h = self.norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        return h, attention_mask.bool()

class TopPPIAblation(nn.Module):

    def __init__(self, variant="full"):
        super().__init__()
        self.variant = normalize_variant(variant)
        variant_cfg = ABLATION_VARIANTS[self.variant]
        self.use_lora = variant_cfg["use_lora"]
        self.graph_mode = variant_cfg["graph_mode"]
        self.use_cross_attn = variant_cfg["use_cross_attn"]

        model_name = resolve_model_name(config.model_name)
        print(f"Loading ESM-2 backbone: {model_name}")
        print(f"Ablation variant: {self.variant} | {variant_cfg['description']}")
        self.backbone = EsmModel.from_pretrained(model_name)

        if getattr(config, "gradient_checkpointing", False):
            self.backbone.gradient_checkpointing_enable()

        for param in self.backbone.parameters():
            param.requires_grad = False

        if self.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=getattr(config, "lora_r", 8),
                lora_alpha=getattr(config, "lora_alpha", 32),
                lora_dropout=getattr(config, "lora_dropout", 0.1),
                target_modules=getattr(config, "lora_target_modules", ["query", "value"]),
            )
            self.backbone = get_peft_model(self.backbone, peft_config)

        hidden_dim = getattr(config, "gat_hidden_dim", 512)
        gat_heads = getattr(config, "gat_heads", 4)
        gat_dropout = getattr(config, "gat_dropout", 0.3)

        if self.graph_mode == "none":
            self.graph_extractor = None
            self.gat_adapter = None
            self.sequence_adapter = SequenceProjectionAdapter(
                in_dim=getattr(config, "embedding_dim", 1280),
                hidden_dim=hidden_dim,
                dropout=gat_dropout,
            )
        else:
            self.graph_extractor = AblationGraphExtractor(
                layer_idx=getattr(config, "contact_head_layer", 29),
                head_indices=getattr(config, "contact_head_indices", [4, 7]),
                threshold=getattr(config, "contact_threshold", 0.1),
                k_neighbors=getattr(config, "knn_k", 10),
                graph_mode=self.graph_mode,
            )
            self.gat_adapter = StructureAwareGATAdapter(
                in_dim=getattr(config, "embedding_dim", 1280),
                hidden_dim=hidden_dim,
                heads=gat_heads,
                dropout=gat_dropout,
            )
            self.sequence_adapter = None

        if self.use_cross_attn:
            cross_heads = getattr(config, "cross_attn_heads", 4)
            self.cross_attn_v2h = CrossAttentionLayer(hidden_dim, num_heads=cross_heads)
            self.cross_attn_h2v = CrossAttentionLayer(hidden_dim, num_heads=cross_heads)
        else:
            self.cross_attn_v2h = None
            self.cross_attn_h2v = None

        self.pool_v = AttentionPooling(hidden_dim)
        self.pool_h = AttentionPooling(hidden_dim)

        fused_dim = hidden_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.LayerNorm(fused_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fused_dim // 2, 1),
        )

        self.projector = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Linear(fused_dim // 2, getattr(config, "projection_dim", 128)),
        )

        self.interaction_proj_v = nn.Linear(hidden_dim, hidden_dim)
        self.interaction_proj_h = nn.Linear(hidden_dim, hidden_dim)

    def forward_single_branch(self, input_ids, attention_mask):
        need_attentions = self.graph_mode != "none"
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=need_attentions,
        )
        sequence_output = outputs.last_hidden_state

        if self.graph_mode == "none":
            return self.sequence_adapter(sequence_output, attention_mask)

        edge_indices = self.graph_extractor(outputs.attentions, sequence_output, attention_mask)
        gat_feat, gat_mask = self.gat_adapter(sequence_output, edge_indices, attention_mask)
        return gat_feat, gat_mask

    def forward(self, virus_inputs, host_inputs):
        v_feat, v_mask = self.forward_single_branch(
            virus_inputs["input_ids"], virus_inputs["attention_mask"]
        )
        h_feat, h_mask = self.forward_single_branch(
            host_inputs["input_ids"], host_inputs["attention_mask"]
        )

        if self.use_cross_attn:
            v_mask_bool = ~v_mask.bool()
            h_mask_bool = ~h_mask.bool()
            v_repr = self.cross_attn_v2h(
                query=v_feat,
                key=h_feat,
                value=h_feat,
                key_padding_mask=h_mask_bool,
            )
            h_repr = self.cross_attn_h2v(
                query=h_feat,
                key=v_feat,
                value=v_feat,
                key_padding_mask=v_mask_bool,
            )
        else:
            v_repr = v_feat
            h_repr = h_feat

        v_global, _ = self.pool_v(v_repr, mask=v_mask.bool())
        h_global, _ = self.pool_h(h_repr, mask=h_mask.bool())

        fused_mult = v_global * h_global
        fused_diff = torch.abs(v_global - h_global)
        fused_repr = torch.cat([v_global, h_global, fused_mult, fused_diff], dim=1)

        logits = self.classifier(fused_repr)
        proj_features = self.projector(fused_repr)

        v_proj = torch.tanh(self.interaction_proj_v(v_repr))
        h_proj = torch.tanh(self.interaction_proj_h(h_repr))
        interaction_map = torch.bmm(v_proj, h_proj.transpose(1, 2))
        combined_mask = torch.bmm(v_mask.unsqueeze(2).float(), h_mask.unsqueeze(1).float())
        interaction_map = interaction_map.masked_fill(combined_mask == 0, -1e4)

        v_scores = torch.softmax(interaction_map.mean(dim=2), dim=1)
        h_scores = torch.softmax(interaction_map.mean(dim=1), dim=1)

        return logits, proj_features, (v_scores, h_scores)

def build_ablation_model(variant="full"):
    variant = normalize_variant(variant)
    if variant == "full":

        return TopPPI()
    return TopPPIAblation(variant=variant)
