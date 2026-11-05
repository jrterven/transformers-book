# Chapter 10. Incremental Decoding and KV Representations

Run the examples from the repository root with NumPy.

~~~sh
python -m pip install -r requirements.txt
python -m examples.ch10_kv.run_examples --output output/ch10.json
python -m unittest discover -s tests -p test_kv.py -v
python -m unittest discover -s tests -p test_latent.py -v
~~~

The program first appends the key and value for *car* to the two positions
already cached for *the red*. The query for *car* reads all three positions.
Its attention weights are about (0.4011, 0.1978, 0.4011), giving a head
output of about (1.6044, 1.0000).

## Compare a full pass with a cached pass

The example starts after the projections. Queries have shape
[B, Hq, Q, Dk], keys have shape [B, Hkv, T, Dk], and values have shape
[B, Hkv, T, Dv]. Q is the number of queries being processed together and
T is the length of the key and value history. Each KV head serves
Hq / Hkv neighboring query heads. The optional Boolean mask broadcasts
to [B, Hq, Q, T], with True for an allowed query–key pair. Use
[B, 1, Q, T] for a pattern that can vary by sequence and is shared across
its query heads. The teaching cache holds a contiguous, unpadded history
starting at position zero for each batch item.

Pass the prompt projections to attention_step with no cache for prefill.
For each later call, supply the new rows and the returned (keys, values)
pair. The cache length determines the absolute query start for the causal
mask. The function returns the output, attention weights, and updated
cache. The output calculation uses float64 while the stored arrays retain
their dtype. The example copies rows with NumPy concatenation, which
keeps the append easy to inspect.

The fixed six-position example compares both single-token and multi-token
appends with a full causal pass for each sharing arrangement. In float64,
four, two, and one KV heads take 768, 384, and 192 bytes, respectively.
Those totals count the two array payloads. The program also calculates
the size at two bytes per scalar.

## Read a latent history

The two-head example compares expanded_attention, which expands each
head's content keys and values, with latent_attention, which transforms
the query and then mixes the latent rows before the value projection.
Both use the same rotated positional terms. The compact cache uses
eight scalars for the latent rows and rotary keys. Expanded content
with the same shared rotary keys would use 20. The fixed arrays let
you compare the two calculations on a CPU without model weights or
a dataset.
