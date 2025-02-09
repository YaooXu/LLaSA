
import torch
from easydict import EasyDict
from torch_geometric.data import Data
import numpy as np

class BipartiteData(Data):
    def __init__(self, edge_index1=None, edge_index2=None, x_s=None, x_t=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_index1 = torch.LongTensor(edge_index1) if edge_index1 else None  # [2, N]
        self.edge_index2 = torch.LongTensor(edge_index2) if edge_index2 else None  # [2, N]

        if isinstance(x_s, list):
            self.x_s = torch.LongTensor(x_s)
        elif isinstance(x_s, np.ndarray):
            self.x_s = torch.Tensor(x_s)
        else:
            self.x_s  = None

        if isinstance(x_t, list):
            self.x_t = torch.LongTensor(x_t)
        elif isinstance(x_t, np.ndarray):
            self.x_t = torch.Tensor(x_t)
        else:
            self.x_t  = None

    def __inc__(self, key, value, *args, **kwargs):
        if key in ["edge_index", "edge_index1", "corr_edge_index", "edge_index_corr1", "edge_index_corr2"]:
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        elif key in ['edge_index2']: # important!
            return torch.tensor([[self.x_t.size(0)], [self.x_s.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

            
class TableConverter:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.data_args = EasyDict(
            {
                "max_token_length": 64,
                "max_row_length": 20,
                "max_column_length": 20,
                "electra": False,
            }
        )

    def _text2table(self, sample):

        smpl = sample.split(HEADER_TAG)
        cap = smpl[0].replace(CAP_TAG, "").strip()
        smpl = smpl[1].split(ROW_TAG)
        headers = [h.strip() for h in smpl[0].strip().split("|")]
        cells = [list(map(lambda x: x.strip(), row.strip().split("|"))) for row in smpl[1:]]
        for row in cells:
            assert len(row) == len(headers)

        return cap, headers, cells

    def _text2graph(self, table, return_dict=False):
        if type(table) is str:
            table = table.replace("col :", "<caption> [TAB] <header>")
            table = re.sub(r"row\s\d+\s:", "<row>", table)
            cap, headers, data = self._text2table(table)
        else:
            if type(table["header"][0]) == list:
                table["header"], table["rows"] = table["header"][0], table["rows"][0]
            cap = ""
            headers, data = table["header"], table["rows"]

            # dummy row
            if len(data) == 0:
                data = [[' '] * len(headers)]

        cap = " ".join(cap.split()[: self.data_args.max_token_length])  # filter too long caption
        header = [" ".join(h.split()[: self.data_args.max_token_length]) for h in headers][: self.data_args.max_column_length]
        data = [row[: self.data_args.max_column_length] for row in data[: self.data_args.max_row_length]]

        assert len(header) <= self.data_args.max_column_length
        assert len(data) <= self.data_args.max_row_length

        s_nodes, t_nodes = [], []
        nodes, edge_index1, edge_index2= [], [], []

        # caption to hyper-edge (t node)
        cap = f"Table Caption: {cap}"
        t_nodes.append(cap)

        # header to hyper-edge (t node)
        for head in header:
            head = f"Header: {head}"
            t_nodes.append(head)

        # row to hyper edge (t node)
        for i in range(len(data)):
            row = f"Row: {i+1}"
            t_nodes.append(row)

        # cell to nodes (s node)
        for row_i, row in enumerate(data):
            for col_i, word in enumerate(row):
                node_id = len(s_nodes)

                word = f"Node: {word}"
                s_nodes.append(word)

                edge_index1.append([node_id, 0])  # connect to table-level hyper-edge
                edge_index1.append([node_id, col_i + 1])  # connect to col-level hyper-edge
                edge_index1.append([node_id, row_i + 1 + len(header)])  # connect to row-level hyper-edge

                edge_index2.append([0, node_id])  # connect to table-level hyper-edge
                edge_index2.append([col_i + 1, node_id])  # connect to col-level hyper-edge
                edge_index2.append([row_i + 1 + len(header), node_id])  # connect to row-level hyper-edge

        wordpieces_xs_all = self.tokenizer(
            s_nodes,
            return_attention_mask=True,
            padding="max_length",
            truncation=True,
            max_length=self.data_args.max_token_length,
        )['input_ids']

        wordpieces_xt_all = self.tokenizer(
            t_nodes,
            return_attention_mask=True,
            padding="max_length",
            truncation=True,
            max_length=self.data_args.max_token_length,
        )['input_ids']
        
        # add label
        # label_ids = torch.zeros((len(header)-1, self.data_args.label_type_num), dtype=torch.float32)
        # assert len(label_ids) == len(labels) == len(header) -1
        col_mask = [0 for i in range(len(wordpieces_xt_all))]
        col_mask[1 : 1 + len(header)] = [1] * len(header)

        edge_index1 = torch.tensor(edge_index1, dtype=torch.long).T
        edge_index2 = torch.tensor(edge_index2, dtype=torch.long).T

        if not return_dict:
            bigraph = BipartiteData(
                edge_index1=edge_index1,
                x_s=torch.LongTensor(wordpieces_xs_all),
                x_t=torch.LongTensor(wordpieces_xt_all),
                num_nodes=len(wordpieces_xs_all),
                num_hyperedges=len(wordpieces_xt_all),
            )
        else:
            bigraph = dict(
                edge_index1=edge_index1.tolist(),
                edge_index2=edge_index2.tolist(),
                x_s=wordpieces_xs_all,
                x_t=wordpieces_xt_all,
                num_nodes=len(wordpieces_xs_all),
                num_hyperedges=len(wordpieces_xt_all),
            )

        return bigraph


class GraphConverter:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.data_args = EasyDict(
            {
                "max_token_length": 64,
                "max_num_nodes": 400
            }
        )
    
    def _kg_tupels2graph(self, kg_tuples, return_dict=False):
        s_nodes, t_nodes = [], []
        # edge_index1 s -> t, edge_index2 t -> s
        edge_index1, edge_index2 = [], []

        cap = "Graph Caption: "
        t_nodes.append(cap)

        name_to_node_id = {}
        for kg_tuple in kg_tuples:
            h, r, t = kg_tuple
            h = f'Node: {h}' if 'Node' not in h else h
            r = f'Relation: {r}' if 'Relation' not in r else r
            t = f'Node: {t}' if 'Relation' not in t else t

            if h not in name_to_node_id:
                if len(s_nodes) >= self.data_args.max_num_nodes:
                    continue

                name_to_node_id[h] = len(s_nodes)
                s_nodes.append(h)

                edge_index1.append([name_to_node_id[h], 0])
                edge_index2.append([0, name_to_node_id[h]])
            h_node_idx = name_to_node_id[h]

            if t not in name_to_node_id:
                if len(s_nodes) >= self.data_args.max_num_nodes:
                    continue

                name_to_node_id[t] = len(s_nodes)
                s_nodes.append(t)

                edge_index1.append([name_to_node_id[t], 0])
                edge_index2.append([0, name_to_node_id[t]])
            t_node_idx = name_to_node_id[t]           

            # always add relation node
            r_node_idx = len(t_nodes)
            t_nodes.append(r)

            # s -> t
            edge_index1.append([h_node_idx, r_node_idx])

            # t -> s
            edge_index2.append([r_node_idx, t_node_idx])

        wordpieces_xs_all = self.tokenizer(
            s_nodes,
            return_attention_mask=True,
            padding="max_length",
            truncation=True,
            max_length=self.data_args.max_token_length,
        )['input_ids']

        wordpieces_xt_all = self.tokenizer(
            t_nodes,
            return_attention_mask=True,
            padding="max_length",
            truncation=True,
            max_length=self.data_args.max_token_length,
        )['input_ids']

        edge_index1 = torch.tensor(edge_index1, dtype=torch.long).T
        edge_index2 = torch.tensor(edge_index2, dtype=torch.long).T

        if not return_dict:
            bigraph = BipartiteData(
                edge_index1=edge_index1,
                x_s=torch.LongTensor(wordpieces_xs_all),
                x_t=torch.LongTensor(wordpieces_xt_all),
                num_nodes=len(wordpieces_xs_all),
                num_hyperedges=len(wordpieces_xt_all),
            )
        else:
            bigraph = dict(
                edge_index1=edge_index1.tolist(),
                edge_index2=edge_index2.tolist(),
                x_s=wordpieces_xs_all,
                x_t=wordpieces_xt_all,
                num_nodes=len(wordpieces_xs_all),
                num_hyperedges=len(wordpieces_xt_all),
            )

        return bigraph