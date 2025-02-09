import json
import torch.nn.functional as F

instruct = (
    "[INST] <<SYS>>\nYou are an AI assistant that specializes in analyzing and reasoning over structured information. "
    "You will be given a task, optionally with some structured knowledge input. Your answer must strictly adhere to the output format, if specified.\n<</SYS>>\n\n"
)

question_pattern = "Answer the following question with the help of the given list of the knowledge graph triples. knowledge graph triples:\n{kg_tuples}\nquestion:{question}"


def pad_2d_tensor(tensor, max_length, pad_value):
    return F.pad(tensor, (0, max_length - tensor.size(0)), value=pad_value)


def print_trainable_params(model):
    trainable_params = 0
    all_param = 0

    for name, param in model.named_parameters():
        num_params = param.numel()

        all_param += num_params
        if param.requires_grad:
            # print(name, param.shape, num_params)
            trainable_params += num_params

    print(f"{trainable_params} / {all_param}, {trainable_params*100/all_param}%")
    return trainable_params, all_param


def convert_kg_tuples_to_str(all_triples):
    kg_tuples = []
    for triple in all_triples:
        kg_tuples.append(" ".join(triple))
    kg_tuples = " | ".join(kg_tuples)
    return kg_tuples


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_jsonl(path):
    datas = []
    with open(path, "r") as f:
        for line in f:
            datas.append(json.loads(line))
    return datas


def write_jsonl(path, samples):
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def df_to_jsonl(df, filename):
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            json_obj = row.to_json()
            f.write(json_obj)
            f.write("\n")
