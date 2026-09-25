# Checkpoints (Sneha — Part 3)

GitHub blocks files over 100 MB. Full `best.pt` is ~108 MB (G+D, float32), so it stays local/Drive.

| File | On git? | What it is |
|---|---|---|
| **`best_infer.pt` (~43 MB)** | **Yes** | Inference-only: `G_AB` + `G_BA` in float16 |
| `best.pt` (~108 MB) | No | Full train checkpoint (generators + discriminators) |
| `best_smoke.pt` | No | Smoke run |

To reload for demo / more preds:

```python
state = torch.load("checkpoints/best_infer.pt", map_location=device, weights_only=True)
G_AB.load_state_dict({k: v.float() for k, v in state["G_AB"].items()})
G_BA.load_state_dict({k: v.float() for k, v in state["G_BA"].items()})
```

(or `.half()` on the model if you keep AMP.)
