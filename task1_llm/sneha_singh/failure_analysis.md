# Part 1 - Generation failure cases (Sneha)

These three cases are from the full GPU run (10 epochs, 100K/10K TinyStories, RTX 5090). Val CE 0.857, perplexity 2.36, top-1 next-char accuracy 72.9%.

## Case 1 - Repetition loop (greedy)
**Snippet (greedy decoding):**
```
...You are very happy. You are very happy. You are very happ...
```
**Observation:** Greedy decoding gets stuck in a short repetition loop. Once the model predicts "You are very happy.", the same context comes back next step, so the same continuation is picked again. Fix: temperature/top-k sampling or a repetition penalty.

## Case 2 - Word duplication (greedy)
**Snippet (greedy decoding):**
```
...She added some nice colors and shows them them to the park.
```
**Observation:** The model emits "them them". Character-level GPTs have no word-level boundary awareness, so once a plausible word is complete, the next-char distribution can still favor re-starting the same word. More training or a bigram repetition penalty would help.

## Case 3 - Loss of coherence (temperature 0.8)
**Snippet (temperature 0.8 sampling):**
```
...She added some nice colors the box and made me sure the door.
She saw an amazing tree, but she was feeling so happy that she did not stop anymore.
```
**Observation:** Individual clauses are grammatical but do not connect to the prompt or each other. Topic drifts colors -> box -> door -> tree with no narrative thread. Expected for a 3.2M param character-level model with a 128-char context window. Fixes: larger context, larger model, or word/BPE tokenization.
