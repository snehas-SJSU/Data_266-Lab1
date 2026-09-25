# Part 1 — Generation failure cases (Sneha)

These three cases are from the **smoke** run (2 epochs, tiny split). I will refresh snippets after the full GPU train (≥10 epochs, 100K / 10K).

## Case 1 — Repetition
**Snippet (greedy):**
```
...she he and the the the the the the t and the the the the t the the...
```
**Observation:** After a short start (`As Mia read the book, she`), greedy decoding locks onto `the` and repeats it. The model knows `the` is common, but not how to keep the story moving.

## Case 2 — Broken grammar / nonsense tokens
**Snippet (temperature 0.8):**
```
...sheeollhete the ng ay tre the thedss band F he vet thaie ly...
```
**Observation:** Words are glued together (`sheeollhete`, `thedss`) and letters look random. Character-level models invent bad spellings when they are under-trained.

## Case 3 — Loss of coherence
**Snippet (temperature 0.8, continued):**
```
...coFheand luleum t. bomy wad hed thery s s and acim o he iln the
```
**Observation:** The prompt was about Mia reading a book. The continuation never comes back to Mia, the book, or a clear event. Capital `F`, broken words, and empty `s s` pieces show the story thread is gone.
