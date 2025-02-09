import torch
import torch.nn.functional as F

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_sentence_embeds(model, tokenizer, input_ids, batch_size=512):
    sentence_embeddings_list = []
    
    # Process input_ids in batches
    for i in range(0, len(input_ids), batch_size):
        input_ids_batch = input_ids[i:i+batch_size]
        attention_mask = torch.ne(input_ids_batch, tokenizer.pad_token_id).to(input_ids_batch.device)

        # Compute token embeddings for the batch
        with torch.no_grad():
            model_output = model(input_ids_batch, attention_mask=attention_mask)

        # Perform pooling on the batch
        sentence_embeddings_batch = mean_pooling(model_output, attention_mask)

        # Normalize embeddings for the batch
        sentence_embeddings_batch = F.normalize(sentence_embeddings_batch, p=2, dim=1)

        # Append the results
        sentence_embeddings_list.append(sentence_embeddings_batch)

    # Concatenate all batch embeddings into a single tensor
    sentence_embeddings = torch.cat(sentence_embeddings_list, dim=0)
    
    return sentence_embeddings

