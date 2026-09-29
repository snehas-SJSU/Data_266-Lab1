# Task 1 — Failure Analysis: Character-Level GPT on TinyStories

**Author:** Ritika Mukesh Neema (seed=6638)

Three real generation failures observed in outputs/generated_samples.txt.

## Case 1: Mode collapse under greedy decoding
All 10 greedy-decoded samples are byte-for-byte identical:
"Once upon a time, there was a little boy named Tim. Tim loved to play with his friends. One day, Tim went to the park with his mom. He saw a big box of the box. Tim was very happy and said, 'Thank you, Tim! You are very happy. You are very happy.' Tim and Sue were surprised. They played together and had lots of fun"

Analysis: Greedy decoding always picks the single highest-probability next character, so once the model settles into a well-trodden path through its learned distribution (the very common TinyStories opening "Once upon a time, there was a little..."), there is zero variation across runs or samples -- the model has one dominant deterministic trajectory it always follows. This also surfaces an internal repetition ("You are very happy" said twice) and a malformed phrase ("a big box of the box"), both signs the model's next-token distribction is peaked but not semantically grounded.

## Case 2: Malformed word / broken grammar under temperature sampling
Sample 5 (temperature_sampling): "...he found a little boy in his hand. He did not know what to do. As he found a small box in the box. The box was designt. It flew down the box..."

Analysis: "designt" is not an English word -- a character-level hallucination where the model generates a plausible-looking suffix pattern without it resolving to real vocabulary. "a little boy in his hand" and "a small box in the box" are also nonsensical containment relations. This shows the model has learned surface-level character statistics (common suffixes, sentence shapes) without reliably tracking semantic/spatial plausibility.

## Case 3: Narrative incoherence / object drift
Sample 6 (temperature_sampling): "...It was a big box. It was so big, red ball. It looked at the bird. The bird. The bird liked it up and said, 'Please what to do. I want to find your toy tower...'"

Analysis: The referenced object drifts within a few words -- box, then red ball, then bird -- with no narrative logic connecting the transformations, and "The bird." appears as a bare repeated sentence fragment. This reflects the model's limited context-tracking at only 4 layers / 128-dim embeddings: it can produce locally fluent phrases but loses track of what object it introduced a sentence earlier.

## Additional observation (generation-script limitation, not a model failure)
Samples 8 and 9 contain a literal <|endoftext|>-style boundary token mid-generation, followed by the start of a new, unrelated story. The model correctly learned the story-boundary signal from training data (stories were joined with a boundary marker during preprocessing), but generate.py does not stop sampling at this token, so unrelated stories get concatenated in the output. This is a fixable script issue (stop generation at the boundary token) rather than evidence of a model quality problem.
