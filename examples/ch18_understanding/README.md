# Chapter 18. Language Understanding

Run the offline examples from the repository root with NumPy.

```sh
python -m pip install -r requirements.txt
python -m examples.ch18_understanding.run_examples --output output/ch18.json
python -m unittest discover -s tests -p test_understanding.py -v
```

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
