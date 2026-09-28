# Part 1 — Generation failure cases (Sneha)

These three are from the full run: 10 epochs, 100,000 / 10,000 windows, RTX 5090. Validation cross-entropy was 0.857, perplexity 2.36, and next-character accuracy about 73%. The prompt was “some changes. She added some nice colors”.

## Case 1 — Repetition loop (greedy)

```
...You are very happy. You are very happy. You are very happ...
```

Greedy decoding gets stuck. Once it writes “You are very happy.”, that same phrase is the new context, so it picks the same next characters again. Temperature or a repetition penalty would break the loop. Greedy will not.

## Case 2 — Doubled word (greedy)

```
...She added some nice colors and shows them them to the park.
```

It prints “them them”. A character model does not know where a word ends. After it finishes a likely word, the next character can still be the start of that same word.

## Case 3 — The story wanders (temperature 0.8)

```
...She added some nice colors the box and made me sure the door.
She saw an amazing tree, but she was feeling so happy that she did not stop anymore.
```

Each piece is almost a sentence, but they do not belong together. Colors turn into a box, then a door, then a tree. I expected some of this. The model is about 3.2 million parameters and it only sees 128 characters at a time.
