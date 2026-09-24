# Part 1 — Generation failure cases (Sneha)

Smoke run (2 epochs, tiny split). I will update snippets after the full GPU train (≥10 epochs).

## Case 1 — Repetition
**Snippet (greedy):**
```
...she he and the the the the the the t and the the the the t the the...
```
**Observation:** After a short real start (`As Mia read the book, she`), greedy decoding locks onto `the` and repeats it. The model learned that `the` is common, but not how to move the story forward. More training on the full 100K split should reduce this.

## Case 2 — Broken grammar / nonsense tokens
**Snippet (temperature 0.8):**
```
...sheeollhete the ng ay tre the thedss band F he vet thaie ly...
```
**Observation:** Words are glued together (`sheeollhete`, `thedss`) and letters look random. Character-level models invent invalid spellings when they are under-trained.

## Case 3 — Loss of coherence
**Snippet (temperature 0.8, continued):**
```
...coFheand luleum t. bomy wad hed thery s s and acim o he iln the
```
**Observation:** The prompt was about Mia reading a book. The continuation never returns to Mia, the book, or a clear event. Capital `F`, broken words, and empty `s s` pieces show the story thread is gone.
