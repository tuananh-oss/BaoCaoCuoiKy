"""Ve do thi top-100 accuracy theo round tu cac file results/*.csv (dua vao bao cao)."""
import glob, os
import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 4))
for f in sorted(glob.glob("results/*.csv")):
    d = pd.read_csv(f)
    plt.plot(d["round"], d["top100_acc"] * 100, label=os.path.basename(f)[:-4])
plt.xlabel("Round / Epoch"); plt.ylabel("Top-100 accuracy (%)")
plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("results/compare.png", dpi=200)

rows = []
for f in sorted(glob.glob("results/*.csv")):
    d = pd.read_csv(f).iloc[-1]
    rows.append([os.path.basename(f)[:-4], round(d.top100_acc * 100, 2),
                 round(d.MB_sent, 1), round(d.MB_raw, 1)])
print(pd.DataFrame(rows, columns=["config", "top100(%)", "MB_sent", "MB_raw"]).to_string(index=False))
