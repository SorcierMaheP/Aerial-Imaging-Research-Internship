# Hello I'm the plot runs.csv guy
# I know this code has a lot of redundancy. Sorry, I'm retarded.
# Enter model name AND csv filename respectively as arguments when running this code.

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import sys
import os
import time

MODEL_NAME = sys.argv[1]
CSV_NAME = sys.argv[2]

df = pd.read_csv(f"./{CSV_NAME}")

xpoints = df["epoch"]
ypoints1 = df["train/box_loss"]
ypoints2 = df["val/box_loss"]
ypoints3 = df["train/cls_loss"]
ypoints4 = df["val/cls_loss"]
ypoints5 = df["train/dfl_loss"]
ypoints6 = df["val/dfl_loss"]
ypoints7 = df["metrics/precision(B)"]
ypoints8 = df["metrics/recall(B)"]
ypoints9 = df["metrics/mAP50(B)"]
ypoints10 = df["metrics/mAP50-95(B)"]


fig = plt.figure(layout="constrained")
fig.suptitle(MODEL_NAME, size="xx-large", weight="800")
gs = GridSpec(2, 12, figure=fig)
ax1 = fig.add_subplot(gs[0, 0:4])
ax2 = fig.add_subplot(gs[0, 4:8])
ax3 = fig.add_subplot(gs[0, 8:12])
ax4 = fig.add_subplot(gs[1, 0:3])
ax5 = fig.add_subplot(gs[1, 3:6])
ax6 = fig.add_subplot(gs[1, 6:9])
ax7 = fig.add_subplot(gs[1, 9:12])

ax1.plot(xpoints, ypoints1, "or-", label="train")
ax1.plot(xpoints, ypoints2, "ob-", label="val")
ax1.set_xlabel("epochs")
ax1.set_ylabel("box_loss")
ax1.legend()

ax2.plot(xpoints, ypoints3, "or-")
ax2.plot(xpoints, ypoints4, "ob-")
ax2.set_xlabel("epochs")
ax2.set_ylabel("cls_loss")

ax3.plot(xpoints, ypoints5, "or-")
ax3.plot(xpoints, ypoints6, "ob-")
ax3.set_xlabel("epochs")
ax3.set_ylabel("dfl_loss")

ax4.plot(xpoints, ypoints7, "og-")
ax4.set_xlabel("epochs")
ax4.set_ylabel("precision(B)")

ax5.plot(xpoints, ypoints8, "og-")
ax5.set_xlabel("epochs")
ax5.set_ylabel("recall(B)")

ax6.plot(xpoints, ypoints9, "og-")
ax6.set_xlabel("epochs")
ax6.set_ylabel("mAP50(B)")

ax7.plot(xpoints, ypoints10, "og-")
ax7.set_xlabel("epochs")
ax7.set_ylabel("mAP50-95(B)")

fig.set_size_inches(18.5, 10.5)
os.makedirs(f"./{MODEL_NAME}", exist_ok=True)
plt.savefig(f"./{MODEL_NAME}/Fig_{int(time.time())}.png")
plt.show()
