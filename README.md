# LLaSA

This is the repository for the paper "[LLaSA: Large Language and Structured Data Assistant](https://arxiv.org/abs/2411.14460)". 

In this documentation, we detail how to construct pretraining datasets and train LLaSA model.
## Install Requirements

Requirements:
- Python 3.10
- Linux
- support for CUDA 12.4

```
pip install -r requirements.txt
```

If you encounter any issues during installing `torch-geometric`, please refer to [torch-geometric](https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html) for manual installation.

## Pretraining

You can also download out pretraining ckpt and skip the pretraining process.

> Due to the accidental deletion of the weight file, we will re-release the weights as soon as possible after retraining and validation.

### Prepare pretraining datasets

```bash
# download pretraining data
git clone https://github.com/YaooXu/TaBERT.git
cd TaBERT
python -m spacy download en_core_web_sm
bash get_pretrain_data.sh

python preprocess/construct_pretrain_data.py 
```

### Pretraining

```
bash pretrain_gformer.sh
```


## Training


```
# download and process data
python preprocess/construct_sft_data.py

# convert all data to hypergraph
python preprocess/convert_table_to_graph_hytrel.py


bash ./train_llasa.sh
```


## Evaluation
```
bash ./predict.sh
```