q = attention.split_heads(attention.q_proj(x))
new_k = attention.split_heads(attention.k_proj(x))
new_v = attention.split_heads(attention.v_proj(x))
if self.self_kv[i] is None:
    k, v = new_k, new_v
else:
    old_k, old_v = self.self_kv[i]
    k = torch.cat((old_k, new_k), dim=2)
    v = torch.cat((old_v, new_v), dim=2)
self.self_kv[i] = (k, v)
attended = _attend(attention, q, k, v)
x = layer.norm1(x + attended)
