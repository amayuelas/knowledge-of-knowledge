# %%
# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")

if os.path.exists("plots") == False:
    os.mkdir("plots")
# %%
# ----------------------------------------------- # 
# --------  Plot models before training --------- #
## F1
f1s = [0.03, 0.01, .04, 0.24, 0.23, .34, .28]
accs = [0.25, 0.15, .22, 0.30, 0.33, .39, .41]
models = ["Llama-2-7B", "Llama-2-13B","Llama-2-70B", "Llama-2-7B Chat", "Llama-2-13B Chat", "GPT-3.5", "GPT-4"]

custom_palette = sns.color_palette("YlOrBr")
plt.barh(models, f1s, color=custom_palette)
plt.xlabel("F1 Score", fontweight='bold')
plt.ylabel("Model", fontweight='bold')
plt.savefig("plots/before_training_f1.png", dpi=300, bbox_inches='tight')

## Accuracy
custom_palette = sns.color_palette("Blues")
plt.barh(models, accs, color=custom_palette)
plt.xlabel("F1 Score", fontweight='bold')
plt.ylabel("Model", fontweight='bold')
plt.savefig("plots/before_training_acc.png", dpi=300, bbox_inches='tight')

# %%
## F1 and Acc
custom_palette = sns.color_palette("colorblind")

fig, ax = plt.subplots()
ax.barh(np.arange(0, len(models)), f1s, height=.35, color=custom_palette, hatch='//')
ax.barh(np.arange(0.3, len(models) + 0.3), accs, height=.35, color=custom_palette)
ax.set_yticks(np.arange(0.3, len(models) + 0.3), models)
plt.xlabel("F1 Score / Acc", fontweight='bold')
plt.ylabel("Model", fontweight='bold')
plt.legend(["F1 Score", "Accuracy"])
plt.savefig("plots/before_training_f1_acc.png", dpi=300, bbox_inches='tight')

# ----------------------------------------------- # 
# ---------  Plot models after training --------- #

# TODO



# %%
## Plot different size models
f1s = [0.03, 0.03, .03, .04, .05, .69, .67]
accs = [.25, .26, .24, .27, .29, .23, .22]
train_size = [0, 32, 64, 128, 256, 512, 1024]

plt.plot(train_size, f1s, linewidth=2)
plt.plot(train_size, accs,  linestyle='--', linewidth=2)
plt.xlabel("Train Size (#Questions)", fontweight='bold')
plt.ylabel("F1 Score / Acc", fontweight='bold')
plt.legend(["F1 Score", "Accuracy"])
plt.savefig("plots/train_size_7b_.png", dpi=300, bbox_inches='tight')
# %%
