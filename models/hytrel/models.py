import torch
from .layers import *
from transformers import AutoTokenizer, AutoModel
from utils.sentence_transformer import get_sentence_embeds


# class Embedding(nn.Module):
#   def __init__(self, config):
#     super(Embedding, self).__init__()
#     self.tok_embed = nn.Embedding(32001, config.hidden_size, padding_idx=config.pad_token_id)  # token embedding
#     self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#     self.dropout = nn.Dropout(config.hidden_dropout_prob)

#   def forward(self, x_s, x_t):
#     embedding_s, embedding_t = self.tok_embed(x_s), self.tok_embed(x_t)
#     embedding_s, embedding_t = torch.div(torch.sum(embedding_s, dim=1), torch.count_nonzero(x_s, dim=1).unsqueeze(-1)), torch.div(torch.sum(embedding_t, dim=1), torch.count_nonzero(x_t, dim=1).unsqueeze(-1))
#     return self.dropout(self.norm(embedding_s)), self.dropout(self.norm(embedding_t))


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        # self.tok_embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)  # token embedding
        # self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.proj = nn.Linear(1024, config.hidden_size)
        self.llm_pad_token_id = config.llm_pad_token_id

        model_path = "sentence-transformers/all-roberta-large-v1"
        self.sbert = AutoModel.from_pretrained(
            model_path,
        )
        self.sbert.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def forward(self, x_s, x_t):
        if x_s.dtype == torch.long:
            x_s = get_sentence_embeds(self.sbert, self.tokenizer, x_s).to(self.proj.weight.dtype)
            x_t = get_sentence_embeds(self.sbert, self.tokenizer, x_t).to(self.proj.weight.dtype)
        else:
            x_s = x_s.to(self.proj.weight.dtype)
            x_t = x_t.to(self.proj.weight.dtype)

        return self.proj(x_s), self.proj(x_t)


class EncoderLayer(nn.Module):
    """SetTransformer Encoder Layer"""

    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob
        self.V2E = AllSetTrans(config=config)
        self.fuse = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.E2V = AllSetTrans(config=config)

    def forward(self, embedding_s, embedding_t, edge_index1, edge_index2):

        # # reverse the index
        if edge_index2 is None:
            reversed_edge_index = torch.stack([edge_index1[1], edge_index1[0]], dim=0)
        else:
            reversed_edge_index = edge_index2
        # from nodes to hyper-edges
        embedding_t_tem = F.relu(self.V2E(embedding_s, edge_index1))

        # from hyper-edges to nodes
        embedding_t = torch.cat([embedding_t, embedding_t_tem], dim=-1)
        # fuse the output t_embeds with original t_embeds, or the t_embeds will not have the original info
        embedding_t = F.dropout(self.fuse(embedding_t), p=self.dropout, training=self.training)
        embedding_s = F.relu(self.E2V(embedding_t, reversed_edge_index))
        embedding_s = F.dropout(embedding_s, p=self.dropout, training=self.training)

        return embedding_s, embedding_t


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.embed_layer = Embedding(config)
        self.layer = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, data, not_pad_graph_embeds=False):
        embedding_s, embedding_t = self.embed_layer(data.x_s, data.x_t)
        # embedding_t: t1, t2, t3, s1, s2, s3, s4
        embedding_t = torch.cat([embedding_t, embedding_s], dim=0)

        # Add self-loop
        num_nodes, num_hyper_edges = data.x_s.size(0), data.x_t.size(0)

        self_edge_index1 = torch.tensor([[i, num_hyper_edges + i] for i in range(num_nodes)]).T
        edge_index1 = torch.cat([data.edge_index1, self_edge_index1.to(data.edge_index1.device)], dim=-1)

        self_edge_index2 = torch.tensor([[num_hyper_edges + i, i] for i in range(num_nodes)]).T
        edge_index2 = torch.cat([data.edge_index2, self_edge_index2.to(data.edge_index2.device)], dim=-1)

        x_s_idxes = data["x_s_ptr"].tolist()

        all_embedding_s = []
        for i, layer_module in enumerate(self.layer):
            embedding_s, embedding_t = layer_module(embedding_s, embedding_t, edge_index1, edge_index2)
            all_embedding_s.append(embedding_s)

        if not_pad_graph_embeds:
            # only last layer
            list_graph_embeds, _ = self.get_list_of_graph_embeds_and_attns(all_embedding_s[-1], x_s_idxes)
            return list_graph_embeds
        else:
            if 'return_all_layer' in self.config and self.config.return_all_layer:
                all_graph_embeds = []
                for embedding_s in all_embedding_s:
                    graph_embeds, graph_attention_mask = self.get_graphs_embeds_and_attns(embedding_s, x_s_idxes)
                    all_graph_embeds.append(graph_embeds)
                # [batch, num_nodes, dim] * num_layers
                graph_embeds = torch.stack(all_graph_embeds, 1)
            else:
                graph_embeds, graph_attention_mask = self.get_graphs_embeds_and_attns(all_embedding_s[-1], x_s_idxes)
                
            return graph_embeds, graph_attention_mask

    def get_list_of_graph_embeds_and_attns(self, embedding_s, x_s_idxes):
        list_graph_attn = []
        list_graph_embeds = []
        for i in range(len(x_s_idxes) - 1):
            graph_embeds = embedding_s[x_s_idxes[i] : x_s_idxes[i + 1], :]  # s_nodes
            list_graph_attn.append(torch.LongTensor([1] * (x_s_idxes[i + 1] - x_s_idxes[i])))
            list_graph_embeds.append(graph_embeds)

        return list_graph_embeds, list_graph_attn

    def get_graphs_embeds_and_attns(self, embedding_s, x_s_idxes):
        list_graph_embeds, list_graph_attn = self.get_list_of_graph_embeds_and_attns(embedding_s, x_s_idxes)

        graph_attention_mask = torch.nn.utils.rnn.pad_sequence(list_graph_attn, batch_first=True).to(embedding_s.device)
        graph_embeds = torch.nn.utils.rnn.pad_sequence(list_graph_embeds, batch_first=True).to(embedding_s.device)

        max_seq_len = 400
        graph_attention_mask = torch.nn.functional.pad(graph_attention_mask, (0, max_seq_len - graph_attention_mask.size(1)))
        graph_embeds = torch.nn.functional.pad(graph_embeds, (0, 0, 0, max_seq_len - graph_embeds.size(1)))

        return graph_embeds, graph_attention_mask


class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """

    def __init__(self, temperature=0.5):
        super().__init__()

        self.temperature = temperature
        self.loss_fct = nn.CrossEntropyLoss()

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """

        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        cos_sim = torch.einsum("id,jd->ij", z_i, z_j) / self.temperature
        labels = torch.arange(cos_sim.size(0)).long().to(proj_1.device)
        loss = self.loss_fct(cos_sim, labels)

        return loss
