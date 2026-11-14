# Chapter 19. Language Understanding

Run the offline examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch19_understanding.run_examples --output output/ch19.json
python -m unittest discover -s tests -p test_understanding.py -v
```

The [measured experiments](PRETRAINED_FIGURES.md) use actual encoder
representations, trained sentence heads, and logits from two released task
checkpoints. The recorded outputs reproduce all six figures offline.

## Pool token features for a sentence prediction

`sentence_logits` accepts contextual features `[time, width]` or
`[batch, time, width]`. The Boolean mask marks the token rows that
contribute to each mean. The head then maps the pooled vector to
one logit per class. Its weight has one row per class. The fixture
uses fixed features and weights so the mean and logits can be
calculated by hand.

## Align one word label to its first piece

The example uses word IDs in reading order, starting at zero. All
pieces of one word have the same ID and form a contiguous run.
Use ID `-1` at special and padding tokens. `first_piece_labels` places
the word label at the start of each run and fills other token rows
with the ignore label, which defaults to `-100`.

`first_piece_predictions` takes one predicted class ID per token
position and reads the first piece of each word. This uses the same
positions as supervision and lets the evaluation compare one tag per
word.

## Compare complete entity spans

`bio_spans` reads `B-type`, `I-type`, and `O` tags. A span is a
triple of zero-based start, exclusive end, and entity type. Strict
decoding raises if an inside tag does not continue a span of the
same type. Optional repair treats that inside tag as the start of
a new span. Gold tags use strict decoding in `span_metrics`. Choose
the prediction policy before evaluating and report it with results.

An exact match needs the same start, end, and type. Span precision,
recall, and F1 return `None` when their respective denominators are
zero. Word accuracy compares the supplied tags and returns `None`
for an empty word sequence. Prediction repair affects span decoding.


## Exercises and complete solutions

The [twelve exercises](EXERCISES.md) progress from hand calculations to code and
evaluation designs. The [solutions guide](../../solutions/ch19_understanding/README.md)
links an executed notebook and two PDF editions, each covering every exercise.

The attention mask controls access within the encoder. The pooling mask selects
output rows for a summary. The loss mask selects supervised predictions.
Changing excluded padding rows in fixed features leaves masked means unchanged;
this does not test whether an encoder incorrectly attended to padded inputs.
An ignored continuation piece can still influence the first piece through attention.
The `-100` target is a loss sentinel, whereas `O` is a real supervised class.

The decoder uses IOB2. Convert IOB1 annotations explicitly before calling it.
Changing a span's boundary or type creates an unmatched prediction and an
unfound gold entity. To calculate dataset micro scores, add per-sentence
match, predicted and gold counts before dividing, or retain sentence IDs in
the span keys. Identical offsets in different sentences are not a match.
An average of sentence F1 scores generally differs from micro F1.

For flat BIO tagging, each word has one label; nested or discontinuous entities
need a different representation. Recover original word indices and declare
how truncated or partially retained words are handled before scoring.
