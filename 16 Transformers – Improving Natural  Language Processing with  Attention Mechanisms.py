import torch
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
print(sentence)
torch.manual_seed(123)
embed = torch.nn.Embedding(10, 16)
embedded_sentence = embed(sentence).detach()
print(embedded_sentence.shape)
#omega = torch.empty(8, 8)
omega = embedded_sentence.matmul(embedded_sentence.T)
import torch.nn.functional as F
attention_weights = F.softmax(omega, dim=1)
print(attention_weights.shape)
print(attention_weights.sum(dim=1))
x_2 = embedded_sentence[1, :]
context_vec_2 = torch.zeros(x_2.shape)
for j in range(8):
    x_j = embedded_sentence[j, :]
    context_vec_2 += attention_weights[1, j] * x_j
print(context_vec_2)
context_vectors = torch.matmul(attention_weights, embedded_sentence)
print(torch.allclose(context_vectors[1], context_vec_2))

torch.manual_seed(123)
d = embedded_sentence.shape[1]
U_query = torch.randn(d, d)
U_key = torch.randn(d, d)
U_value = torch.randn(d, d)

x_2 = embedded_sentence[1]
query_2 = U_query.matmul(x_2)
key_2 = U_key.matmul(x_2)
value_2 = U_value.matmul(x_2)
keys = U_key.matmul(embedded_sentence.T).T
values = U_value.matmul(embedded_sentence.T).T
print(torch.allclose(keys[1], key_2))
print(torch.allclose(values[1], value_2))
omega_23 = query_2.dot(keys[2])
print(omega_23)
omega_2 = query_2.matmul(keys.T)
print(omega_2)
attention_weights_2 = F.softmax(omega_2 / d**0.5, dim=0)
print(attention_weights_2)
context_vectors_2 = attention_weights_2.matmul(values)
print(context_vectors_2)

torch.manual_seed(123)
d = embedded_sentence.shape[1]
one_U_query = torch.rand(d, d)
h = 8
multihead_U_query = torch.rand(h, d, d)
multihead_U_key = torch.rand(h, d, d)
multihead_U_value = torch.rand(h, d, d)
multihead_query_2 = multihead_U_query.matmul(x_2) #qj = uqj * x
print(multihead_U_query.shape)
multihead_key_2 = multihead_U_key.matmul(x_2) #kj = ukj * x
multihead_value_2 = multihead_U_value.matmul(x_2) #vk = uvj * x
print(multihead_key_2[2])
stacked_inputs = embedded_sentence.T.repeat(8, 1, 1)
print(stacked_inputs.shape)
multihead_keys = torch.bmm(multihead_U_key, stacked_inputs) #k ==> h=8
print(multihead_keys.shape)
multihead_keys = multihead_keys.permute(0, 2, 1)
print(multihead_keys.shape)
print(multihead_keys[2, 1])
multihead_values = torch.matmul(multihead_U_value, stacked_inputs)
multihead_values = multihead_values.permute(0, 2, 1)