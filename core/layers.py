import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
try:
    from torch_geometric.nn import GATv2Conv
    _HAS_GATV2 = True
except ImportError:
    GATv2Conv = None
    _HAS_GATV2 = False
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch, remove_self_loops

class ImplicitContactMapExtractor(nn.Module):

    def __init__(self, layer_idx, head_indices, threshold=0.1, k_neighbors=10):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_indices = head_indices
        self.threshold = threshold
        self.k = k_neighbors

    def forward(self, attention_outputs, sequence_embeddings, attention_mask):

        layer_attentions = attention_outputs[self.layer_idx]
        batch_size, _, seq_len, _ = layer_attentions.size()

        edge_indices_list = []

        for b in range(batch_size):
            valid_len = attention_mask[b].sum().item()

            selected_heads = layer_attentions[b, self.head_indices, :valid_len, :valid_len]
            contact_map = selected_heads.mean(dim=0)
            mask_contact = contact_map > self.threshold
            mask_contact.fill_diagonal_(0)
            edge_index_contact = mask_contact.nonzero().t()

            curr_emb = sequence_embeddings[b, :valid_len, :]

            dist_matrix = torch.cdist(curr_emb, curr_emb)

            k_val = min(self.k + 1, valid_len)
            _, indices = dist_matrix.topk(k=k_val, dim=1, largest=False)

            source_nodes = torch.arange(valid_len, device=curr_emb.device).unsqueeze(1).expand(-1, k_val)
            edge_index_knn = torch.stack([source_nodes.flatten(), indices.flatten()], dim=0)

            edge_index = torch.cat([edge_index_contact, edge_index_knn], dim=1)

            edge_index, _ = remove_self_loops(edge_index)
            edge_index = torch.unique(edge_index, dim=1)

            edge_indices_list.append(edge_index)

        return edge_indices_list

class StructureAwareGATAdapter(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim,
        heads=4,
        dropout=0.3,
        num_layers=2,
        conv_type="gat",
        use_residual=False,
        use_gate=False,
        gate_init_bias=-2.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_gate = use_gate

        conv_type = conv_type.lower()
        if conv_type == "gatv2" and _HAS_GATV2:
            Conv = GATv2Conv
        else:
            Conv = GATConv

        if num_layers == 1:
            self.gat_layer1 = Conv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=False)
            self.gat_layer2 = None
        else:
            self.gat_layer1 = Conv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
            self.gat_layer2 = Conv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)

        self.extra_layers = nn.ModuleList(
            Conv(hidden_dim, hidden_dim, heads=1, dropout=dropout, concat=False)
            for _ in range(max(0, num_layers - 2))
        )

        if self.use_residual or self.use_gate:
            self.skip_proj = nn.Linear(in_dim, hidden_dim)
        else:
            self.skip_proj = None

        if self.use_gate:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.gate[0].weight)
            nn.init.constant_(self.gate[0].bias, gate_init_bias)
        else:
            self.gate = None

        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ELU()

    def forward(self, x, edge_indices_list, batch_mask):
        batch_size, seq_len, _ = x.size()
        data_list = []

        for i in range(batch_size):
            valid_len = batch_mask[i].sum().item()
            node_features = x[i, :valid_len, :]
            edges = edge_indices_list[i]
            data_list.append(Data(x=node_features, edge_index=edges))

        batch_data = Batch.from_data_list(data_list).to(x.device)

        input_nodes = batch_data.x
        h = self.gat_layer1(input_nodes, batch_data.edge_index)
        if self.gat_layer2 is not None:
            h = self.activation(h)
            h = self.gat_layer2(h, batch_data.edge_index)

        for extra_layer in self.extra_layers:
            h = self.activation(h)
            h = extra_layer(h, batch_data.edge_index)

        if self.use_gate:
            skip = self.skip_proj(input_nodes)
            gate = self.gate(torch.cat([skip, h], dim=-1))
            h = gate * h + (1.0 - gate) * skip
        elif self.use_residual:
            h = h + self.skip_proj(input_nodes)

        dense_h, mask = to_dense_batch(h, batch_data.batch, batch_size=batch_size, max_num_nodes=seq_len)

        dense_h = self.norm(dense_h)

        return dense_h, mask

class CrossAttentionLayer(nn.Module):

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, query, key, value, key_padding_mask=None):

        attn_out, _ = self.mha(query, key, value, key_padding_mask=key_padding_mask)

        x = self.norm1(query + self.dropout(attn_out))
        x_ffn = self.ffn(x)
        x = self.norm2(x + x_ffn)

        return x

class AttentionPooling(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.attn_linear = nn.Linear(in_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, x, mask=None):

        scores = self.attn_linear(self.tanh(x))

        if mask is not None:

            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e4)

        weights = F.softmax(scores, dim=1)

        pooled = torch.sum(weights * x, dim=1)

        return pooled, weights
