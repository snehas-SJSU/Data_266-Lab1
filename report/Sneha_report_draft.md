# DATA266 Lab 1 — Team 5 Report (draft)

**Course:** DATA266  
**Team:** 5  
**Members:** Sneha Singh · Ritika Mukesh Neema  
**Repo:** https://github.com/snehas-SJSU/Data_266-Lab1  

> Draft for the final PDF. **Sneha’s sections** are filled from the repo.  
> **Ritika’s sections** are empty placeholders.  
> Kaggle public score is still TBD on my side. Part 1 full run is in.

---

## 0. How we split the work

| Part | Sneha Singh | Ritika Mukesh Neema |
|---|---|---|
| 1 — GPT (TinyStories) | Own model + full GPU run done | Own model (pending) |
| 2 — Yelp polarity | 3 models, full run done | Own 3 models (pending) |
| 3 — CycleGAN | Train + local eval + audit (my column) | Own train + audit column (pending) |
| Report / Drive / Kaggle | My half + my Drive / CSV | Her half + her Drive / CSV |

Shared: raw datasets under `task*/data/`. Each person only commits under their own `sneha_singh/` or `ritika_mukesh_neema/` folder.

---

# Part 1 — GPT from scratch

## 1.A Sneha Singh

### Goal
Train a small GPT from scratch on TinyStories (character-level). No `nn.Transformer`, no prebuilt attention blocks.

### Architecture (mine)
- Char tokenizer (`char_to_idx` / `idx_to_char`)
- 4 layers, 4 heads, emb 256, block size 128
- Causal triu mask
- Weight tying + AdamW (no decay on bias / LayerNorm)

### Training
| Setting | Full run (reported) |
|---|---|
| Split | 100000 train / 10000 val |
| Epochs | 10 |
| Device | NVIDIA GeForce RTX 5090, AMP |
| Config | `smoke: false` |
| Time | 457 s, peak memory 349 MB |

### Metrics

| Metric | Value |
|---|---|
| Train CE | 0.912 |
| Val CE | 0.857 |
| Perplexity | 2.36 |
| Top-1 next-char acc | 0.729 |

Full columns: `task1_llm/sneha_singh/metrics_report.csv`.

Figures: `task1_llm/sneha_singh/outputs/loss_curves.png`, `outputs/samples.txt`.

### Failure cases
1. **Repetition** — greedy decoding loops on “You are very happy.”
2. **Word duplication** — greedy emits “them them”.
3. **Lost thread** — temperature 0.8 drifts from colors to a box, a door, then a tree.

Details: `task1_llm/sneha_singh/failure_analysis.md`.

---

## 1.B Ritika Mukesh Neema

_Placeholder — Ritika fills architecture, hardware, metrics, samples, and 3 failure cases from `task1_llm/ritika_mukesh_neema/`._

---

## 1.C Comparison (after both finish)

| | Sneha | Ritika |
|---|---|---|
| Layers / heads / emb | 4 / 4 / 256 | _add_ |
| Val CE / PPL | 0.857 / 2.36 | _add_ |
| Notes | | |

---

# Part 2 — Yelp polarity sentiment

## 2.A Sneha Singh

### Goal
Three models on Yelp polarity (not IMDB). Embeddings from scratch — no pretrained LMs.

### Models
1. **Baseline** — mean pool over embeddings + linear  
2. **Experimental A** — BiLSTM  
3. **Experimental B** — TextCNN (filters 3/4/5)

### Setup
- Preprocess: lower, strip punct, stopwords, lemmatize  
- Vocab from train only; 100k sampled, 99,989 after empty-review drop / 10k / 10k; 5 epochs; batch 64; emb 100; max_len 128  
- Device: Apple M4, 16 GB unified memory, MPS (`smoke: false`). Same machine for all three models.

### Test results (mine)

| Model | Acc | Macro-F1 | ROC-AUC | MCC | Brier | ECE | Train time |
|---|---|---|---|---|---|---|---|
| Baseline | 0.923 | 0.923 | 0.973 | 0.845 | 0.059 | 0.011 | 114 s |
| BiLSTM (A) | 0.918 | 0.918 | 0.975 | 0.837 | 0.060 | 0.017 | 1706 s |
| TextCNN (B) | 0.921 | 0.921 | 0.975 | 0.841 | 0.061 | 0.024 | 108 s |

Best macro-F1: **baseline**. McNemar vs baseline: BiLSTM p=0.091, TextCNN p=0.424. Neither is clearly different from the baseline on this test set.

**Figures:**  
- `task2_sentiment/sneha_singh/outputs/eda.png`  
- `…/baseline_cm_full.png`, `experimental_a_cm_full.png`, `experimental_b_cm_full.png`

### Errors (baseline)
I logged 20 mistakes on the new baseline. Biggest themes:
- Very short blurbs the model reads as positive (`food always good`, `good beer`)
- Negation and mixed reviews (`not restaurant closed`, coffee that was warm but not good)
- Long reviews where the label and the wording pull in different directions

Full table: `task2_sentiment/sneha_singh/failure_analysis.md`.

### Honest limits
100k of the 560k Yelp train split, not the whole corpus. max_len 128 cuts long reviews. Negation words are kept, and mixed reviews are still hard. Peak memory is 0.0 in the CSV because MPS does not fill the CUDA peak counter. I did not guess a number.

---

## 2.B Ritika Mukesh Neema

_Placeholder — Ritika fills her own 3 models + metrics + error review from `task2_sentiment/ritika_mukesh_neema/`._

---

## 2.C Comparison (after both finish)

| | Sneha best | Ritika best |
|---|---|---|
| Model | Baseline (macro-F1 0.923) | _add_ |
| Acc / F1 | 0.923 / 0.923 | _add_ |
| Takeaway | Mean-pool still leads. BiLSTM and TextCNN are close, and McNemar does not separate them from the baseline. | |

---

# Part 3 — CycleGAN (photo ↔ Monet)

## 3.A Sneha Singh

### Goal
Unpaired CycleGAN. Kaggle direction is **photo → Monet (A2B)**. No pretrained image models on the outputs — only my generators.

### Architecture
- G: ResNet-9 @256, reflection pad, instance norm, tanh  
- D: PatchGAN, LSGAN (MSE)  
- Losses: adv + cycle (λ=10) + identity (0.5λ) + image pool (50)

### Train
- CUDA GPU, AMP, batch 4  
- 40 constant + 40 decay epochs (~4.3 h, peak mem ~6568 MB)  
- nearest upsample, label smoothing 0.9, ~800 photos / epoch; full photo set for A2B inference (7038)

### Local metrics (`evaluate_local.py`)

| Direction | FID | KID | MiFID | LPIPS | content cos | cycle L1 |
|---|---|---|---|---|---|---|
| A2B (Kaggle) | 89.99 | 0.018 | 0.403 | 0.452 | 0.532 | 0.112 |
| B2A | 92.23 | 0.030 | 0.423 | 0.365 | 0.586 | 0.103 |

Same A2B FID/MiFID are in `task3_gan/sneha_singh/submission.csv` for Kaggle.

### Kaggle (mine — fill after upload)

| | |
|---|---|
| File | `submission.csv` (not `images.zip`) |
| Self-reported | FID 89.99, MiFID 0.403 |
| Public score | _TBD_ |
| Private / rank | _TBD_ |

### Figures
- `task3_gan/sneha_singh/outputs/samples/grid.png`  
- `task3_gan/sneha_singh/outputs/loss_curves/losses.png`  
- A few audit examples from `outputs/human_audit/` (e.g. good blossom/lighthouse vs noisy 01455 / 04853)

### What goes wrong (my eyes)
Grain / checkerboard in sky and water is the main issue. Layout of the photo usually stays. Colors can wash out. Discriminator stayed very low (~0.2) the whole training — that may be why textures look noisy. I did **not** flag obvious memorized Monet copies on my 30-sample pass (all mem = N).

### Human audit (my column)
- Sheet: `outputs/human_audit/audit_30.csv`  
- My averages (rough): style ~1.4, content ~1.7, artifacts ~1.4  
- Inter-rater κ: _after Ritika rates the same 30_

### If I retrain
More epochs or slightly higher identity; better upsampling for checkerboard.

---

## 3.B Ritika Mukesh Neema

_Placeholder — Ritika fills architecture, train hardware, metrics, `submission.csv` scores, failures, and her audit column from `task3_gan/ritika_mukesh_neema/`._

---

## 3.C Comparison + joint audit (after both finish)

| | Sneha | Ritika |
|---|---|---|
| A2B FID / MiFID (local) | 89.99 / 0.403 | _add_ |
| Kaggle public | _TBD_ | _add_ |
| Audit κ (same 30) | _after both rate_ | |

---

# Closing (team)

- Individual folders on git hold the real notebooks, CSVs, and plots.  
- Part 1 `best.pt` (12 MB) and Part 2 checkpoints stay on git. Part 3 `best.pt` (113 MB) goes on Drive — link in root README.  
- This draft becomes `DATA266_Lab1_Report_Team_5.pdf` once Kaggle scores and Ritika’s sections are in.

**Sneha — still open before PDF is final**
1. Upload Kaggle CSV → paste public score here  
2. Drive `best.pt` link in README  
3. Merge Ritika’s write-ups + κ into one PDF with her  

---

*Tone note: written in first person for my sections. Ritika should write hers the same way in the placeholders above.*
