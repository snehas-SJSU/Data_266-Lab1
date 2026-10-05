"""Run the TA's evaluation notebook (Part3_Evaluation_Script.ipynb) code cells unchanged.

Only the four folder paths are overridden. Usage:
    python ta_score.py [--gen_a2b DIR] [--gen_b2a DIR]
Defaults score Sneha's current outputs in task3_gan/sneha_singh/outputs.
Writes submission.csv into the current working directory (as the TA's last cell does).
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]   # task3_gan/

ap = argparse.ArgumentParser()
ap.add_argument("--gen_a2b", default=str(REPO / "sneha_singh" / "outputs" / "pred_A2B"))
ap.add_argument("--gen_b2a", default=str(REPO / "sneha_singh" / "outputs" / "pred_B2A"))
ap.add_argument("--notebook", default=str(REPO / "Part3_Evaluation_Script.ipynb"), help="the course evaluation notebook")
args = ap.parse_args()

nb = json.load(open(args.notebook, encoding="utf-8"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

g = {"__name__": "__main__"}
for i, src in enumerate(cells):
    if src.strip().startswith("# !pip"):
        continue
    exec(compile(src, f"<TA cell {i}>", "exec"), g)
    if "BASE =" in src:  # config cell: point the four folders at our files
        g["REAL_MONET"] = str(REPO / "data" / "monet_jpg")
        g["REAL_PHOTO"] = str(REPO / "data" / "photo_jpg")
        g["GEN_A2B"] = args.gen_a2b
        g["GEN_B2A"] = args.gen_b2a

fid_a2b, fid_b2a = g["fid_A2B"], g["fid_B2A"]
mifid_a2b, mifid_b2a = g["mifid_A2B"], g["mifid_B2A"]
sub_fid, sub_mifid = (fid_a2b + fid_b2a) / 2, (mifid_a2b + mifid_b2a) / 2
print(f"\nSCORE = (FID + MiFID) / 2 = {(sub_fid + sub_mifid) / 2:.4f}   [FID {sub_fid:.3f}, MiFID {sub_mifid:.4f}]")
