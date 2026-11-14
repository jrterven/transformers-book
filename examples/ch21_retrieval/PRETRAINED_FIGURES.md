# Chapter 21 model demonstrations

Run these commands from the repository root. The figure script uses the
saved run and needs NumPy and Matplotlib.

```sh
python -m pip install -r requirements.txt matplotlib==3.11.2
python -m examples.ch21_retrieval.pretrained_figures
```

To run the checkpoints, use Python 3.11 or newer. The first run fetches
about 4.13 GB of Safetensors weights and the tokenizer files. Add
`--offline` on subsequent runs to read from the cache. The script
verifies the published weight hashes and uses the built-in model classes.
The saved run used four CPU threads, float32, and greedy generation.

```sh
python -m pip install -r requirements-ch21-figures.txt
python -m examples.ch21_retrieval.pretrained_observations --output output/ch21-pretrained.json
python -m examples.ch21_retrieval.pretrained_figures --record output/ch21-pretrained.json
```

The archive and questions are fictional. Eight questions have evidence
IDs and two ask for facts absent from the collection. The recorded model
revisions are [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3),
[Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/tree/e61197ed45024b0ed8a2d74b80b4d909f1255473),
and [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17).

## Follow one question

1. BM25 scores lowercase word runs, including numbers. The dense model
   pools its last visible token and normalizes the vector. Compare the
   actual scores of P01 and P02 for the renewal paraphrase.
2. Fuse up to five positive BM25 results and five dense results with
   reciprocal rank fusion. Score up to six candidates with the reranker.
   Its relevance score is the softmax over the `no` and `yes` logits.
3. Pack whole passages by counting the complete chat template after each
   trial addition. The regular cap is 360 input tokens and the answer
   reserve is 64. The tight cap fits the first lexical hit by itself.
   The chat includes one example of a complete answer with a citation.
4. Inspect the emitted text and its citations. The passage P02 supports
   the date it states. The question asks for a location. The oracle runs
   provide P01 directly, compare two prompt formats, and change just the
   location to test whether the answer follows the edit.

The run stores lexical and dense scores, the two relevance logits for
each candidate, every packing decision, the generated token IDs, and
the evidence recall by stage. The manual answer audit checks whether
the requested facts are complete and supported by the cited passages.
The scores describe this small archive and these settings.
[Campus notices](CAMPUS_NOTICES.json) · [Model outputs](PRETRAINED_RUN.json) ·
[Answer audit](ANSWER_AUDIT.json).
