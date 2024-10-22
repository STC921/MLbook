import torch
import torch.nn.functional as F

torch.manual_seed(123)
sentence = torch.tensor(
    [0,
     7,
     1,
     2,
     5,
     6,
     4,
     3]
)
embed = torch.nn.Embedding(10, 16)
embedded_sentence = embed(sentence)
# print(embedded_sentence.shape)
sequence_length, d = embedded_sentence.shape
h = 6
d_k, d_v = d, d
multihead_U_query = torch.rand(h, d, d_k)
multihead_U_key = torch.rand(h, d, d_k)
multihead_U_value = torch.rand(h, d, d_v)

stacked_input = embedded_sentence.T.repeat(h, 1, 1)
multihead_queries = torch.bmm(multihead_U_query, stacked_input)
multihead_keys = torch.bmm(multihead_U_key, stacked_input)
multihead_values = torch.bmm(multihead_U_value, stacked_input)
print(multihead_U_query.shape)
print(multihead_U_key.shape)
print(multihead_U_value.shape)
multihead_queries = multihead_queries.permute(0, 2, 1)
multihead_keys = multihead_keys.permute(0, 2, 1)
print(multihead_U_query.shape)
print(multihead_U_key.shape)
print(multihead_U_value.shape)

#Step 1
dot_product = torch.bmm(multihead_queries.transpose(1, 2), multihead_keys)
scaled_dot_product = dot_product / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
print(scaled_dot_product.shape)
#Step 2
attention_weights = F.softmax(scaled_dot_product, dim=-1)
print(attention_weights.shape)
#Step 3
multihead_outputs = torch.bmm(attention_weights, multihead_values)
print(multihead_outputs.shape)
#Step 4
multihead_outputs = multihead_outputs.permute(1, 0, 2).contiguous().view(sequence_length, h * d_v)
print(multihead_outputs.shape)
