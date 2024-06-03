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
# -----  Plot dataset 1st-word statistics ------- #

df = pd.read_json("data/knowledge-of-knowledge/knowns_unknowns.jsonl", lines=True)
df["question_first_word"] = df["question"].apply(lambda x: str(x).lower().split(" ")[0])

# Create a figure and a grid of subplots
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Plot the first plot
df_plot = pd.DataFrame(df[df["source"]=="turk"]["question_first_word"].value_counts()[:10])
df_plot.plot.pie(y="count", ax=axes[0], ylabel="", fontsize=16)
axes[0].set_title("Crowd-source", fontsize=21)
axes[0].legend(bbox_to_anchor=(1.25, -0.05), ncol=3, fontsize=16)


# Plot the second plot
df_plot = pd.DataFrame(df[df["source"]=="squad"]["question_first_word"].value_counts()[:10])
df_plot.plot.pie(y="count", ax=axes[1], ylabel="", fontsize=16)
axes[1].set_title("SQuAD",fontsize=21)
axes[1].legend(bbox_to_anchor=(1.25, -0.05), ncol=3, fontsize=16)

# Plot the third plot
df_plot = pd.DataFrame(df[df["source"]=="triviaqa"]["question_first_word"].value_counts()[:10])
df_plot.plot.pie(y="count", ax=axes[2], ylabel="", fontsize=16)
axes[2].set_title("TriviaQA", fontsize=21)
axes[2].legend(bbox_to_anchor=(1.25, -0.05), ncol=3, fontsize=16)

# Plot the fourth plot
df_plot = pd.DataFrame(df[df["source"]=="hotpotqa"]["question_first_word"].value_counts()[:10])
df_plot.plot.pie(y="count", ax=axes[3], ylabel="", fontsize=16)
axes[3].set_title("HotPotQA",fontsize=21)
axes[3].legend(bbox_to_anchor=(1.25, -0.05), ncol=3, fontsize=16)

# Show the figure
plt.tight_layout()
plt.savefig("plots/first_word.png", dpi=300, bbox_inches='tight')



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

# %%
# ----------------------------------------------- # 
# ---------  Plot models after comparison --------- #

f1s_before = [0.03, 0.01, .04, 0.24, 0.23, .32, .34, .28]
accs_before = [0.25, 0.15, .22, 0.30, 0.33, .39, .39, .41]
models = ["Llama-2-7B", "Llama-2-13B","Llama-2-70B", "Llama-2-7B Chat", "Llama-2-13B Chat", "Llama-2-70B Chat", "GPT-3.5", "GPT-4"]

f1s_after = [.49, .72, 0, .75, .74, 0, 0, 0]
accs_after = [.21, .24, 0, .21, .22, 0, 0, 0]

# F1s
custom_palette = sns.color_palette("colorblind")
fig, ax = plt.subplots(figsize=(6,5))
ax.barh(np.arange(0, len(models)), f1s_before, height=.45, color=custom_palette)
ax.barh(np.arange(0.45, len(models) + 0.45), f1s_after, height=.45, color=custom_palette, hatch='//')
ax.set_yticks(np.arange(0.45, len(models) + 0.45), models, fontsize=18)
plt.xticks(fontsize=18)
plt.xlabel("F1 Score", fontweight='bold', fontsize=20)
# plt.ylabel("Model", fontweight='bold', fontsize=16)
plt.legend(["Original", "Finetuned"], fontsize=18)
plt.savefig("plots/f1_score_before_after", bbox_inches='tight')


custom_palette = sns.color_palette("colorblind")
fig, ax = plt.subplots(figsize=(6,5))
ax.barh(np.arange(0, len(models)), accs_before, height=.45, color=custom_palette)
ax.barh(np.arange(0.45, len(models) + 0.45), accs_after, height=.45, color=custom_palette, hatch='//')
ax.set_yticks(np.arange(0.45, len(models) + 0.45), models)
ax.set_yticks(np.arange(0.45, len(models) + 0.45), models, fontsize=18)
plt.xticks(fontsize=18)
plt.xlim(0,0.7)
plt.xlabel("Known Question Accuracy", fontweight='bold', fontsize=18)
# plt.ylabel("Model", fontweight='bold', fontsize=16)
plt.legend(["Original", "Finetuned"], fontsize=20)
plt.savefig("plots/acc_score_before_after", bbox_inches='tight')



# %%
# ----------------------------------------------- # 
# ---------  Plot train size --------- #
# Lllama-2 7b
f1s = [0.03, 0.03, .03, .04, .05, .69, .67]
accs = [.25, .26, .24, .27, .29, .23, .22]
train_size = [0, 32, 64, 128, 256, 512, 1024]

fig = plt.figure()
plt.plot(train_size, f1s, linewidth=8)
# plt.plot(train_size, accs,  linestyle='--', linewidth=8)
plt.xlabel("Train Size (#Questions)", fontweight='bold', fontsize=30)
plt.ylabel("F1 Score", fontweight='bold', fontsize=30)
# plt.legend(["F1 Score"], fontsize=30)
plt.tight_layout()
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.grid(linewidth=5)
plt.savefig("plots/f1_train_size_7b.png", dpi=300, bbox_inches='tight')



# Llama-2 7b-Chat
f1s = [.24, .26, .25, .21, .18, .57, .76]
accs = [.30, .31, .31, .32, .29, .22, .22]
train_size = [0, 32, 64, 128, 256, 512, 1024]

fig = plt.figure()
plt.plot(train_size, f1s, linewidth=8)
# plt.plot(train_size, accs,  linestyle='--', linewidth=8)
plt.xlabel("Train Size (#Questions)", fontweight='bold', fontsize=30)
plt.ylabel("F1 Score", fontweight='bold', fontsize=30)
# plt.legend(["F1 Score", "Accuracy"], fontsize=16)
plt.tight_layout()
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.grid(linewidth=5)
plt.savefig("plots/f1_train_size_7b_chat.png", dpi=300, bbox_inches='tight')


# Llama-2 13b
f1s = [.01, .02, .03, .04, .06, .73, .73]
accs = [.15, .17, .16, .21, .24, .24, .28]
train_size = [0, 32, 64, 128, 256, 512, 1024]

fig = plt.figure()
plt.plot(train_size, f1s, linewidth=8)
# plt.plot(train_size, accs,  linestyle='--', linewidth=8)
plt.xlabel("Train Size (#Questions)", fontweight='bold', fontsize=30)
plt.ylabel("F1 Score", fontweight='bold', fontsize=30)
# plt.legend(["F1 Score", "Accuracy"], fontsize=16)
plt.tight_layout()
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.grid(linewidth=5)
plt.savefig("plots/f1_train_size_13b.png",dpi=300, bbox_inches='tight')


# Llama-2 13b-chat
f1s = [.23, .20, .20, .20, .08, .58, .71]
accs = [.33, .34, .33, .33, .30, .25, .25]
train_size = [0, 32, 64, 128, 256, 512, 1024]

fig = plt.figure()
plt.plot(train_size, f1s, linewidth=8)
# plt.plot(train_size, accs,  linestyle='--', linewidth=8)
plt.xlabel("Train Size (#Questions)", fontweight='bold', fontsize=30)
plt.ylabel("F1 Score", fontweight='bold', fontsize=30)
# plt.legend(["F1 Score", "Accuracy"], fontsize=16)
plt.tight_layout()
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.grid(linewidth=5)
plt.savefig("plots/f1_train_size_13b_chat.png", dpi=300, bbox_inches='tight')


# %%
# --------------- ACCURACY PLOTS ------------------ #
# Llama 7b
f1s = [0.03, 0.03, .03, .04, .05, .69, .67]
accs = [.25, .26, .24, .27, .29, .23, .22]
train_size = [0, 32, 64, 128, 256, 512, 1024]

fig = plt.figure()
# plt.plot(train_size, f1s, linewidth=8)
plt.plot(train_size, accs, linewidth=8, color='tab:orange')
plt.xlabel("Train Size (#Questions)", fontweight='bold', fontsize=30)
plt.ylabel("Accuracy", fontweight='bold', fontsize=30)
# plt.legend(["Accuracy"], fontsize=30)
plt.tight_layout()
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.grid(linewidth=5)
plt.savefig("plots/acc_train_size_7b.png", dpi=300, bbox_inches='tight')



# Llama-2 7b-Chat
f1s = [.24, .26, .25, .21, .18, .57, .76]
accs = [.30, .31, .31, .32, .29, .22, .22]
train_size = [0, 32, 64, 128, 256, 512, 1024]

fig = plt.figure()
# plt.plot(train_size, f1s, linewidth=8)
plt.plot(train_size, accs, linewidth=8, color='tab:orange')
plt.xlabel("Train Size (#Questions)", fontweight='bold', fontsize=30)
plt.ylabel("Accuracy", fontweight='bold', fontsize=30)
# plt.legend(["F1 Score", "Accuracy"], fontsize=16)
plt.tight_layout()
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.grid(linewidth=5)
plt.savefig("plots/acc_train_size_7b_chat.png", dpi=300, bbox_inches='tight')


# Llama-2 13b
f1s = [.01, .02, .03, .01, .02, .73, .74]
accs = [.15, .16, .15, .21, .23, .25, .26]
train_size = [0, 32, 64, 128, 256, 512, 1024]

fig = plt.figure()
# plt.plot(train_size, f1s, linewidth=8)
plt.plot(train_size, accs, linewidth=8, color='tab:orange')
plt.xlabel("Train Size (#Questions)", fontweight='bold', fontsize=30)
plt.ylabel("Accuracy", fontweight='bold', fontsize=30)
# plt.legend(["F1 Score", "Accuracy"], fontsize=16)
plt.tight_layout()
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.grid(linewidth=5)
plt.savefig("plots/acc_train_size_13b.png",dpi=300, bbox_inches='tight')


# Llama-2 13b-chat
f1s = [.23, .20, .20, .20, .08, .58, .71]
accs = [.33, .34, .33, .33, .30, .25, .25]
train_size = [0, 32, 64, 128, 256, 512, 1024]

fig = plt.figure()
# plt.plot(train_size, f1s, linewidth=8)
plt.plot(train_size, accs, linewidth=8, color='tab:orange')
plt.xlabel("Train Size (#Questions)", fontweight='bold', fontsize=30)
plt.ylabel("Accuracy", fontweight='bold', fontsize=30)
# plt.legend(["Accuracy"], fontsize=16)
plt.tight_layout()
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.grid(linewidth=5)
plt.savefig("plots/acc_train_size_13b_chat.png", dpi=300, bbox_inches='tight')

# %%
# ------------------ CATEGORIES ------------------- #
# ----------------------------------------------- #
# --------------  Plot categories --------------- #

labels = ["future", "unsolved", "controversial", "false assump.", "counterfactual", "ambiguous"]
precisions_7b       = [0.73, 0.44, 0.60, 0.36, 0.81, 0.17]
precisions_7b_chat  = [0.73, 0.41, 0.70, 0.22, 0.90, 0.24]
precisions_13b      = [0.77, 0.45, 0.59, 0.63, 0.90, 0.25]
precisions_13b_chat = [0.9, 0.67, 0.62, 0.74, 0.89, 0.38]
recalls_7b        = [0.40, 0.08, 0.07, 0.29, 0.25, 0.08]
recalls_7b_chat   = [0.60, 0.18, 0.59, 0.59, 0.57, 0.15]
recalls_13b       = [0.52, 0.39, 0.69, 0.44, 0.68, 0.22]
recalls_13b_chat  = [0.35, 0.24, 0.62, 0.33, 0.59, 0.21]
f1s_7b        = [0.52, 0.14, 0.13, 0.32, 0.38, 0.11]
f1s_7b_chat   = [0.66, 0.25, 0.64, 0.32, 0.70, 0.19]
f1s_13b       = [0.62, 0.42, 0.64, 0.52, 0.77, 0.26]
f1s_13b_chat  = [0.49, 0.35, 0.62, 0.45, 0.71, 0.27]

# Set up bar positions
plt.figure(figsize=(10, 7))
bar_width = 0.2
r1 = np.arange(len(labels))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]


plt.bar(r1, f1s_7b, color='b', width=bar_width, edgecolor='white', label='7b')
plt.bar(r2, f1s_7b_chat, color='g', width=bar_width, edgecolor='white', label='7b-chat')
plt.bar(r3, f1s_13b, color='orange', width=bar_width, edgecolor='white', label='13b')
plt.bar(r4, f1s_13b_chat, color='r', width=bar_width, edgecolor='white', label='13b-chat')
for tick in plt.yticks()[0]:
    plt.axhline(y=tick, color='gray', linestyle='-', linewidth=0.5, zorder=0)
plt.xticks([r + bar_width for r in range(len(labels))], labels, fontsize=18, rotation=25, ha='right')
plt.ylabel('F1 Score', fontsize=18, fontweight='bold')
# plt.title('F1 Score of Different Models', fontsize=16)
plt.grid(axis='y', linestyle='-', linewidth=2)
plt.grid(axis='x', linestyle='-', linewidth=0)
plt.ylim(0,.8)
plt.yticks(fontsize=18)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize=18, ncol=4)
plt.savefig("plots/f1_categories.png", dpi=300, bbox_inches='tight')
# %%
# Comparing training all categories vs specific categories
# Using llama-7b-chat, without simcse, f1 scores only
# Specific category trained on 1:1 ratio

category_f1_dict = {"future unknown": 0.71, "unsolved problem": 0.55, "controversial": 0.80, "false assumption": 0.59, "counterfactual": 0.53, "ambiguous": 0.29}
kok_all_f1_dict  = {"future unknown": 0.66, "unsolved problem": 0.25, "controversial": 0.64, "false assumption": 0.32, "counterfactual": 0.70, "ambiguous": 0.19}

labels = ["future", "unsolved", "controversial", "false assump.", "counterfactual", "underspecified"]
category_f1s = list(category_f1_dict.values())
kok_all_f1s = list(kok_all_f1_dict.values())

# Set up bar positions
bar_width = 0.4
r1 = np.arange(len(labels))
r2 = [x + bar_width for x in r1]

# Bar plot for category f1
plt.figure(figsize=(9, 6))
plt.bar(r1, category_f1s, color='r', width=bar_width, edgecolor='white', label='Individual Category')
plt.bar(r2, kok_all_f1s, color='b', width=bar_width, edgecolor='white', label='All Categories')

# Add horizontal lines at each y-axis tick
for tick in plt.yticks()[0]:
    plt.axhline(y=tick, color='gray', linestyle='-', linewidth=0.5, zorder=0)

# Annotate each bar with their value
for i, (d1, d2) in enumerate(zip(category_f1s, kok_all_f1s)):
    plt.text(i, d1 + 0.03, d1, ha='center', fontsize=18)
    plt.text(i + bar_width, d2 + 0.03, d2, ha='center', fontsize=18)

# Set labels and title
plt.ylabel('F1 Score', fontsize=17, fontweight='bold')
plt.xticks([r + bar_width for r in range(len(labels))], labels, fontsize=18, rotation=21, ha='right')
plt.yticks(fontsize=18)
plt.legend(loc='upper left', bbox_to_anchor=(0, 1.2), ncols=2, fontsize=18)
plt.grid(axis='y', linestyle='-', linewidth=2)
plt.grid(axis='x', linestyle='-', linewidth=0)
plt.tight_layout()
plt.savefig("plots/f1_categories_specific_vs_all.png", dpi=300, bbox_inches='tight')
# %%
# ----------------------------------------------- #
# -------------  MultiAgent Debate -------------- #
llama7bchat_accs = {
    "MMLU": 0.28,
    "CSQA": 0.37, 
    "ARC": 0.31, 
    "Chess": 0.07
}

llama7bchat_stds = {
    "MMLU": 0.04582575695,
    "CSQA": 0.02886751346, 
    "ARC": 0.04366157731, 
    "Chess": 0.009574271078
}

finetuned7bchat_accs = {
    "MMLU": 0.39,
    "CSQA": 0.46, 
    "ARC": 0.41,
    "Chess": 0.13
}

finetuned7bchat_stds = {
    "MMLU": 0.07,
    "CSQA": 0.02309401077, 
    "ARC": 0.01867520638,
    "Chess": 0.02217355783
}

# plot bar chart with both accs
labels = list(llama7bchat_accs.keys())
llama7bchat_accs = list(llama7bchat_accs.values())
llama7bchat_stds = list(llama7bchat_stds.values())
finetuned7bchat_accs = list(finetuned7bchat_accs.values())
finetuned7bchat_stds = list(finetuned7bchat_stds.values())

# Set up bar positions
plt.figure(figsize=(6, 5))
bar_width = 0.35
r1 = np.arange(len(labels))
r2 = [x + bar_width for x in r1]

plt.bar(r1, llama7bchat_accs, width=bar_width, edgecolor='white', label='Llama-7b-chat')
plt.errorbar(r1,llama7bchat_accs, yerr=llama7bchat_stds, fmt='none', ecolor='black', capsize=5)
plt.bar(r2, finetuned7bchat_accs, color='r',  width=bar_width, edgecolor='white', label='Finetuned-7b-chat')
plt.errorbar(r2,finetuned7bchat_accs, yerr=finetuned7bchat_stds, fmt='none', ecolor='black', capsize=5)


# Add horizontal lines at each y-axis tick
for tick in plt.yticks()[0]:
    plt.axhline(y=tick, color='gray', linestyle='-', linewidth=0.5, zorder=0)

# Annotate each bar with their value
for i, (d1, d2) in enumerate(zip(llama7bchat_accs, finetuned7bchat_accs)):
    plt.text(i-0.05, (d1 + llama7bchat_stds[i])+0.01, f'{d1:.2f}', ha='center', fontsize=17, weight='bold')
    plt.text(i + bar_width-0.05, (d2 + finetuned7bchat_stds[i])+0.01, f'{d2:.2f}', ha='center', fontsize=17, weight='bold')

# Set labels and title
# plt.xlabel('Category')
plt.ylabel('Accuracy', fontsize=17, fontweight='bold')
plt.xticks([r + bar_width for r in range(len(labels))], labels, fontsize=18, rotation=0)
plt.yticks(fontsize=18)
plt.legend(loc='upper left', bbox_to_anchor=(-.2, 1.2), ncols=2, fontsize=18)
plt.grid(axis='y', linestyle='-', linewidth=2)
plt.grid(axis='x', linestyle='-', linewidth=0)
plt.savefig("plots/acc_multiagent.png", dpi=300, bbox_inches='tight')


# %%
# Plot known question accuracy for above
labels_acc = ["7b", "7b-chat", "13b", "13b-chat"]
accs = [0.21, 0.25, 0.27, 0.24]

plt.figure(figsize=(6, 5))
plt.bar(labels_acc, accs, color=['b','g','orange','r'], width=0.5, edgecolor='white')
for tick in plt.yticks()[0]:
    plt.axhline(y=tick, color='gray', linestyle='-', linewidth=0.5, zorder=0)
for i, value in enumerate(accs):
    plt.text(i, value + 0.008, str(value), ha='center', va='bottom', fontsize=18)
plt.ylabel('Known Question Accuracy', fontsize=21)
plt.yticks(fontsize=18)
plt.xticks([r for r in range(len(labels_acc))], labels_acc, fontsize=18)
plt.savefig("plots/acc_known_questions_categories.png", dpi=300, bbox_inches='tight')
# %%

## Plot Precision VS Recall
data = {
    "GPT-4":{
        "precision": [0.5329754601, 0.6357738647, 0.7116827438, 0.75, 0.7586872587, 0.7641509434, 0.797979798, 0.8214285714, 0.9666666667],
        "recall": [1, 0.9870503597, 0.9553956835, 0.8158273381, 0.5654676259, 0.3496402878, 0.2273381295, 0.09928057554, 0.04172661871],
        "FPR": [0.8929618768,0.5762463343,0.3944281525,0.2771260997,0.1832844575,0.1099706745,0.05865102639,0.0219941349,0.00146627566]
    },

    "GPT-3.5": {
        "precision": [0.5095307918, 0.5795644891, 0.6743027888, 0.7279693487, 0.7450199203, 0.7705479452, 0.8465608466, 0.9069767442, 1],
        "recall": [1, 0.9956834532, 0.9741007194, 0.8201438849, 0.5381294964, 0.3237410072, 0.2302158273, 0.1683453237, 0.1107913669 ],
        "FPR": [0.9809384164,0.7360703812,0.4794721408,0.3123167155,0.1876832845,0.09824046921,0.04252199413,0.01759530792,0]
    },

    "Llama 70B-Chat": {
        "precision": [0.5054545455, 0.5237377543, 0.5913419913, 0.6764346764, 0.7207207207, 0.8181818182, 0.9047619048, 0.9174311927, 0.9493670886],
        "recall": [1, 1, 0.9827338129, 0.7971223022, 0.4604316547, 0.2978417266, 0.218705036, 0.1438848921, 0.1079136691],
        "FPR": [0.9970674487,0.926686217,0.6920821114,0.3885630499,0.1818181818,0.06744868035,0.02346041056,0.01319648094, 0.005865102639]
    },

    "Llama 13B-Chat": {
        # "precision": [0.5050872093, 0.5095588235, 0.5443037975, 0.5802469136, 0.5301794454, 0.5043478261, 0.8487394958, 0.9113924051, 0.962962963],
        # "recall": [1, 0.9966942149, 0.9971223022, 0.8115107914, 0.4676258993, 0.2503597122, 0.145323741, 0.1035971223, 0.07482014388],
        # "FPR": [0.9985337243,0.9780058651,0.8445747801,0.5982404692,0.42228739,0.2507331378,0.02639296188,0.01026392962,0.00293255132]

        "precision": [0.5047272727,0.5092114959,0.5391304348,0.5799180328,0.5389830508,0.5136778116,0.8403361345,0.8933333333,0.9583333333],
        "recall": [0.9985611511,0.9942446043,0.981294964,0.8143884892,0.4575539568,0.2431654676,0.1438848921,0.0964028777,0.06618705036],
        "FPR": [0.9985337243,0.9765395894,0.8548387097,0.6011730205,0.3988269795,0.2346041056,0.02785923754,0.01173020528,0.00293255132]
    },
    
    "Llama 7B-Chat": {
        "precision": [0.5061998541, 0.5246398787, 0.5835475578, 0.6843467012, 0.7786458333, 0.8497409326, 0.9256198347, 0.9240506329, 0.9365079365],
        "recall": [0.9985611511,0.9956834532,0.9798561151,0.7611510791,0.4302158273,0.235971223,0.1611510791,0.1050359712,0.08489208633],
        "FPR": [0.9926686217,0.9193548387,0.7126099707,0.357771261,0.1246334311,0.04252199413,0.01319648094,0.008797653959,0.005865102639]
    },

    "Llama-70B": {
        "precision": [0.5047204067,0.5066371681,0.5241871531,0.5282511211,0.5206703911,0.5138211382,0.9,1,1],
        "recall": [1,0.9884892086,0.9510791367,0.8474820144,0.6705035971,0.454676259,0.02589928058,0.01007194245,0.005755395683],
        "FPR": [1,0.9809384164,0.8797653959,0.7712609971,0.6290322581,0.4384164223,0.00293255132,0,0]
    },

    "Llama-13B": {
        "precision": [0.5050872093,0.5066469719,0.5103448276,0.5189393939,0.5148367953,0.494,0.8666666667,1,1],
        "recall": [1,0.9870503597,0.9582733813,0.5913669065,0.4992805755,0.3553956835,0.01870503597,0.007194244604,0.001438848921],
        "FPR": [0.9985337243,0.9794721408,0.9369501466,0.5586510264,0.4794721408,0.3709677419,0.00293255132,0,0]
    },

    "Llama-7B": {
        "precision": [0.5050946143,0.5071590053,0.5547169811,0.5756630265,0.5149700599,0.5649350649,0.9565217391,1,1],
        "recall": [0.9985611511,0.9683453237,0.8460431655,0.5309352518,0.2474820144,0.1251798561,0.03165467626,0.01007194245,0.001438848921],
        "FPR": [0.9970674487,0.9589442815,0.6920821114,0.3988269795,0.2375366569,0.09824046921,0.00146627566,0,0]
    },

    "Finetuned-13B-Chat": {
        "precision": [0.507703595,0.5376518219,0.6007905138,0.730506156,0.8416370107,0.8955823293,0.9156118143,0.9336283186,0.9410430839],
        "recall": [0.9956834532, 0.9553956835,0.8748201439,0.7683453237,0.6805755396,0.6417266187,0.6244604317,0.6071942446,0.5971223022],
        "FPR":[0.9838709677,0.8372434018,0.5923753666,0.288856305,0.1304985337,0.07624633431,0.05865102639,0.04398826979,0.03812316716]
    },

    "Finetuned-7B-Chat": {
        "precision": [0.5054704595,0.5202177294,0.5650301464,0.6185258964,0.6586169045,0.676201373,0.6823529412,0.694002448,0.7061118336],
        "recall": [0.9971223022,0.9625899281,0.9438848921,0.8935251799,0.8633093525,0.8503597122,0.8345323741,0.8158273381,0.781294964],
        "FPR": [0.9941348974,0.9046920821,0.7404692082,0.5615835777,0.4560117302,0.4149560117,0.3958944282,0.366568915,0.3313782991]
    },

    "Finetuned-13B": {
        "precision": [0.5091844232,0.5414920369,0.625,0.7288888889,0.8076923077,0.8521400778,0.8714285714,0.8757763975,0.8967741935],
        "recall": [0.9971223022,0.9294964029,0.8273381295,0.7079136691,0.6647482014,0.6302158273,0.6143884892,0.6086330935,0.6],
        "FPR": [0.9794721408,0.8020527859,0.5058651026,0.2683284457,0.1612903226,0.1114369501,0.09237536657,0.08797653959,0.07038123167]
    },

    "Finetuned-7B": {
        "precision": [0.5069699193,0.5435897436,0.6152897657,0.7050359712,0.7846534653,0.818452381,0.8406779661,0.875,0.8986784141],
        "recall": [0.992816092,0.9137931034,0.716954023,0.5632183908,0.4554597701,0.3951149425,0.3563218391,0.3318965517,0.2931034483],
        "FPR": [0.9824561404,0.7807017544,0.4561403509,0.2397660819,0.1271929825,0.08918128655,0.06871345029,0.04824561404,0.03362573099]
    },

}

models = list(data.keys())
models = [
    "Llama 13B-Chat", "Llama 7B-Chat", "Llama-13B", "Llama-7B",
    "Finetuned-13B-Chat", "Finetuned-7B-Chat", "Finetuned-13B", "Finetuned-7B",
    ]

# Set up bar positions
plt.figure(figsize=(10, 7))
for model in models: 
    marker  = 's' if "Finetuned" in model else 'x'
    linestyle = '-' if "Finetuned" in model else '--'
    if model == "GPT-4" or model == "GPT-3.5":
        linestyle = ':'
    if model == "Llama-13B" or model == "Finetuned-13B":
        color = "tab:orange"
    elif model == "Llama-7B" or model == "Finetuned-7B":
        color = "tab:blue"
    elif model == "Llama 13B-Chat" or model == "Finetuned-13B-Chat":
        color = "tab:red"
    elif model == "Llama 7B-Chat" or model == "Finetuned-7B-Chat":
        color = "tab:green"
    elif model == "Llama 70B":
        color = "tab:olive"
    elif model == "Llama 70B-Chat":
        color = "tab:brown"
    elif model == "GPT-4":
        color = "black"
    elif model == "GPT-3.5":
        color = "tab:pink"
    else:
        # any other color
        color = "tab:purple"    
    plt.plot(data[model]["recall"], data[model]["precision"], linewidth=3, marker=marker, linestyle=linestyle, markersize=10, color=color)
# plot perfect precision-recall curve
plt.plot([0,1], [1,1], linestyle='--', linewidth=3, color='grey')
plt.plot([1,1], [1,0.5], linestyle='--', linewidth=3, color='grey')

plt.xlabel('Recall', fontsize=18, fontweight='bold')
plt.ylabel('Precision', fontsize=18, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(models + ["Perfect classifier"], fontsize=18, bbox_to_anchor=(.95, -.2), ncol=2)
plt.savefig('plots/precision-recall.png', dpi=300, bbox_inches='tight')
# %%
# Make the ROC AUC Plot

plt.figure(figsize=(9, 7))
for model in models: 
    marker  = 's' if "Finetuned" in model else 'o'
    linestyle = '-' if "Finetuned" in model else '--'
    if model == "GPT-4" or model == "GPT-3.5":
        linestyle = ':'
    if model == "Llama-13B" or model == "Finetuned-13B":
        color = "tab:orange"
    elif model == "Llama-7B" or model == "Finetuned-7B":
        color = "tab:blue"
    elif model == "Llama 13B-Chat" or model == "Finetuned-13B-Chat":
        color = "tab:red"
    elif model == "Llama 7B-Chat" or model == "Finetuned-7B-Chat":
        color = "tab:green"
    elif model == "Llama 70B":
        color = "tab:olive"
    elif model == "Llama 70B-Chat":
        color = "tab:brown"
    elif model == "GPT-4":
        color = "black"
    elif model == "GPT-3.5":
        color = "tab:pink"
    else:
        # any other color
        color = "tab:purple"  

    plt.plot(data[model]["FPR"], data[model]["recall"], linewidth=3, marker=marker, linestyle=linestyle, markersize=10, color=color)
    print(model)
# plot perfect ROC curve
plt.plot([0,1], [1,1], linestyle='--', linewidth=3, color='grey')
plt.plot([0,0], [0,1], linestyle='--', linewidth=3, color='grey')

plt.plot([0,1], [0,1], linestyle=':', linewidth=3, color='grey')

# plt.title('ROC Curve', fontsize=21, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=21, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=21, fontweight='bold')
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.legend(models + ["Perfect classifier", "", "Random Classifier"], fontsize=20, bbox_to_anchor=(1.05, -.12), ncol=2)
plt.savefig('plots/ROC.png', dpi=300, bbox_inches='tight')

# %%
# Compute AUC
from sklearn.metrics import auc
for model in models:
    print(model, ": ", auc(data[model]["FPR"], data[model]["recall"]))
# %%
# COMPUTE EER
import numpy as np
from scipy.interpolate import interp1d
for model in models:

    tpr = np.array(data[model]["recall"])
    fpr = np.array(data[model]["FPR"])
    precision = np.array(data[model]["precision"])

    # Calculate the Equal Error Rate (EER)
    # The EER is the point where the FPR and FNR (1-TPR) are equal.
    # This can be approximated by finding the point where FPR and (1-TPR) are closest.
    differences = np.abs(fpr - (1 - tpr))
    min_difference_index = np.argmin(differences)
    eer = (fpr[min_difference_index] + (1 - tpr[min_difference_index])) / 2
    print(model, ": ", eer, " | ", "FPR: ", fpr[min_difference_index], " | ", "TPR: ", tpr[min_difference_index], " | ", "Precision: ", precision[min_difference_index])
# %%
# Compute EER with interpolation
for model in models:

    tpr = np.array(data[model]["recall"])
    fpr = np.array(data[model]["FPR"])
    precision = np.array(data[model]["precision"])

    # Interpolate the TPR and FPR data for a finer resolution
    fpr_min, fpr_max = np.min(fpr), np.max(fpr)
    fpr_interp = np.linspace(fpr_min, fpr_max, 1000)  # Generating points within the range of provided FPR

    tpr_function = interp1d(fpr, tpr, kind='linear')  # Creating a linear interpolation function for TPR
    tpr_interp = tpr_function(fpr_interp)  # Interpolating TPR based on the interpolated FPR

    # Interpolate the precision and recall data for a finer resolution
    recall_min, recall_max = np.min(tpr), np.max(tpr)
    recall_interp = np.linspace(recall_min, recall_max, 1000)  # Generating points within the range of provided recall

    precision_function = interp1d(tpr, precision, kind='linear')  # Creating a linear interpolation function for precision
    precision_interp = precision_function(recall_interp)  # Interpolating precision based on the interpolated recall

    # Compute the differences between FPR and (1 - TPR) for the interpolated data
    differences_interp = np.abs(fpr_interp - (1 - tpr_interp))

    # Find the index of the minimum difference in the interpolated data
    min_difference_index_interp = np.argmin(differences_interp)

    # The EER is approximately the average of the FPR and TPR at this index in the interpolated data
    eer_interp = (fpr_interp[min_difference_index_interp] + (1 - tpr_interp[min_difference_index_interp])) / 2

    # eer_interp, min_difference_index_interp, fpr_interp[min_difference_index_interp], tpr_interp[min_difference_index_interp]

    print(model, ": ", "EER: ", eer_interp, " | FPR: ", fpr_interp[min_difference_index_interp], " | TPR: ", tpr_interp[min_difference_index_interp], " | Precision: ", precision_interp[min_difference_index_interp],
          "| F1: ", 2 * (precision_interp[min_difference_index_interp] * tpr_interp[min_difference_index_interp]) / (precision_interp[min_difference_index_interp] + tpr_interp[min_difference_index_interp]))

    # # Plot the interpolated data
    # plt.figure(figsize=(10, 7))
    # plt.plot(fpr_interp, tpr_interp, linewidth=3, color='blue')
    # plt.plot(fpr, tpr, linewidth=3, color='red')
    # plt.plot([0, 1], [0, 1], linestyle='--', linewidth=3, color='grey')
    # plt.plot([0, 1], [1, 1], linestyle=':', linewidth=3, color='grey')
    # plt.plot([1, 1], [0, 1], linestyle=':', linewidth=3, color='grey')
    # plt.title('ROC Curve', fontsize=18, fontweight='bold')
    # plt.xlabel('False Positive Rate', fontsize=18, fontweight='bold')
    # plt.ylabel('True Positive Rate', fontsize=18, fontweight='bold')
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.legend([model, "no-inter", "Perfect classifier", "", "Random Classifier"], fontsize=18, bbox_to_anchor=(.95, -.2), ncol=2)

# %%

# %%
# ----------------- SELFAWARE ----------------- #

selfaware_data = {

    "Llama 13B-Chat": {
        "precision": [0.3060921248,0.3075075075,0.3145033323,0.3550987505,0.3772241993,0.4011916584,0.7322580645,0.7777777778,0.8571428571],
        "recall": [0.9980620155,0.992248062,0.9602713178,0.8536821705,0.6162790698,0.3914728682,0.2199612403,0.1492248062,0.09302325581],
        "FPR": [0.999144202,0.9867351305,0.9242618742,0.6846384253,0.4492939666,0.2580231065,0.03551561831,0.0188275567,0.006846384253]
           },
    
    "Llama 7B-Chat": {
        "precision": [0.3078302451,0.3164050477,0.3463532665,0.4317953862,0.5382585752,0.6402640264,0.7188328912,0.7783018868,0.8611111111],
        "recall": [0.9980620155,0.996124031,0.9709302326,0.8343023256,0.5930232558,0.3759689922,0.2625968992,0.1598837209,0.09011627907],
        "FPR": [0.9910141207,0.9503637142,0.8091570389,0.4848095849,0.2246469833,0.09328198545,0.04535729568,0.02011125374,0.006418485237]
    },

    "Llama-13B": {
        "precision": [0.3062982769,0.3057553957,0.3116640747,0.3268032057,0.3031557165,0.2424667134,0.7021276596,0.7666666667,0.8333333333],
        "recall": [0.9990310078,0.988372093,0.9709302326,0.7112403101,0.5678294574,0.3352713178,0.03197674419,0.02228682171,0.01453488372],
        "FPR": [0.999144202,0.9910141207,0.946940522,0.6469833119,0.5763799743,0.4625588361,0.005990586222,0.002995293111,0.001283697047]
    },

    "Llama-7B": {
        "precision": [0.3071385723,0.309649395,0.3264391589,0.3646277857,0.3817397556,0.3591836735,0.6305732484,0.6811594203,0.7407407407],
        "recall": [0.992248062,0.9670542636,0.9176356589,0.7451550388,0.5145348837,0.2558139535,0.09593023256,0.04554263566,0.01937984496],
        "FPR": [0.9884467266,0.9520753102,0.8361146769,0.5733846812,0.3679931536,0.2015404365,0.02481814292,0.009413778348,0.002995293111]
    },

    "Finetuned-13B-Chat": {
        "precision": [0.3084056237,0.3241630759,0.3648325359,0.4569247546,0.6055257099,0.6962142198,0.7411167513,0.7624190065,0.7809307605],
        "recall": [0.9990310078,0.9476744186,0.886627907,0.8120155039,0.7645348837,0.730620155,0.7073643411,0.6841085271,0.6666666667],
        "FPR":[0.9893025246,0.8724860933,0.6816431322,0.4261874198,0.2199400941,0.1407787762,0.109114249,0.09413778348,0.08258451006]
    },

    "Finetuned-7B-Chat": {
        "precision": [0.3072792363,0.3147798742,0.3384030418,0.3821501014,0.4294536817,0.4539845758,0.4644012945,0.4900221729,0.5015271839],
        "recall": [0.9980620155,0.9699612403,0.9486434109,0.9127906977,0.8759689922,0.855620155,0.8343023256,0.8246268657,0.7955426357],
        "FPR": [0.9978513107,0.9323919555,0.8189987163,0.6516902011,0.513906718,0.4544287548,0.4249037227,0.3936670946,0.3491655969]
    },

    "Finetuned-13B": {
        "precision": [0.3091018324,0.3336791148,0.3956144358,0.4974842767,0.6185654008,0.6479647965,0.7197860963,0.7320837927,0.7609467456],
        "recall": [0.9970930233,0.9350775194,0.8391472868,0.7664728682,0.7102713178,0.6319742489,0.6521317829,0.6434108527,0.6230620155],
        "FPR": [0.9841677364,0.8245614035,0.5661103979,0.3418913136,0.1934103552,0.1369276851,0.1121095421,0.1039794608,0.0864356012]
    },

    "Finetuned-7B": {
        "precision": [0.3048192771,0.3263671195,0.3871287129,0.4816176471,0.5768421053,0.6464516129,0.6840536513,0.724137931,0.7631067961],
        "recall": [0.980620155,0.9079457364,0.757751938,0.6346899225,0.5310077519,0.4854651163,0.4447674419,0.4069767442,0.3808139535],
        "FPR": [0.9875909285,0.8275566966,0.5297389816,0.3016688062,0.1720154044,0.1172443303,0.09071459136,0.06846384253,0.05220367993]
    },
}


# %%
models = list(selfaware_data.keys())
models = [
    "Llama 13B-Chat", "Llama 7B-Chat", "Llama-13B", "Llama-7B",
    "Finetuned-13B-Chat", "Finetuned-7B-Chat", "Finetuned-13B", "Finetuned-7B", 
    ]

# Set up bar positions
plt.figure(figsize=(10, 7))
for model in models: 
    marker  = 's' if "Finetuned" in model else 'x'
    linestyle = '-' if "Finetuned" in model else '--'
    if model == "GPT-4" or model == "GPT-3.5":
        linestyle = ':'
    if model == "Llama-13B" or model == "Finetuned-13B":
        color = "tab:orange"
    elif model == "Llama-7B" or model == "Finetuned-7B":
        color = "tab:blue"
    elif model == "Llama 13B-Chat" or model == "Finetuned-13B-Chat":
        color = "tab:red"
    elif model == "Llama 7B-Chat" or model == "Finetuned-7B-Chat":
        color = "tab:green"
    elif model == "Llama 70B":
        color = "tab:olive"
    elif model == "Llama 70B-Chat":
        color = "tab:brown"
    elif model == "GPT-4":
        color = "black"
    elif model == "GPT-3.5":
        color = "tab:pink"
    else:
        # any other color
        color = "tab:purple"    
    plt.plot(selfaware_data[model]["recall"], selfaware_data[model]["precision"], linewidth=3, marker=marker, linestyle=linestyle, markersize=10, color=color)
# plot perfect precision-recall curve
plt.plot([0,1], [1,1], linestyle='--', linewidth=3, color='grey')
plt.plot([1,1], [1,0.5], linestyle='--', linewidth=3, color='grey')

plt.xlabel('Recall', fontsize=18, fontweight='bold')
plt.ylabel('Precision', fontsize=18, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(models + ["Perfect classifier"], fontsize=18, bbox_to_anchor=(.95, -.2), ncol=2)
plt.show()
# %%
# Make the ROC AUC Plot

plt.figure(figsize=(9, 7))
for model in models: 
    marker  = 's' if "Finetuned" in model else 'o'
    linestyle = '-' if "Finetuned" in model else '--'
    if model == "GPT-4" or model == "GPT-3.5":
        linestyle = ':'
    if model == "Llama-13B" or model == "Finetuned-13B":
        color = "tab:orange"
    elif model == "Llama-7B" or model == "Finetuned-7B":
        color = "tab:blue"
    elif model == "Llama 13B-Chat" or model == "Finetuned-13B-Chat":
        color = "tab:red"
    elif model == "Llama 7B-Chat" or model == "Finetuned-7B-Chat":
        color = "tab:green"
    elif model == "Llama 70B":
        color = "tab:olive"
    elif model == "Llama 70B-Chat":
        color = "tab:brown"
    elif model == "GPT-4":
        color = "black"
    elif model == "GPT-3.5":
        color = "tab:pink"
    else:
        # any other color
        color = "tab:purple"  

    plt.plot(selfaware_data[model]["FPR"], selfaware_data[model]["recall"], linewidth=3, marker=marker, linestyle=linestyle, markersize=10, color=color)
    print(model)
# plot perfect ROC curve
plt.plot([0,1], [1,1], linestyle='--', linewidth=3, color='grey')
plt.plot([0,0], [0,1], linestyle='--', linewidth=3, color='grey')

plt.plot([0,1], [0,1], linestyle=':', linewidth=3, color='grey')

# plt.title('ROC Curve', fontsize=21, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=21, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=21, fontweight='bold')
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.legend(models + ["Perfect classifier", "", "Random Classifier"], fontsize=20, bbox_to_anchor=(1.05, -.12), ncol=2)
plt.show()
# %%
# COMPUTE EER
import numpy as np
from scipy.interpolate import interp1d
for model in models:

    tpr = np.array(selfaware_data[model]["recall"])
    fpr = np.array(selfaware_data[model]["FPR"])
    precision = np.array(selfaware_data[model]["precision"])

    # Calculate the Equal Error Rate (EER)
    # The EER is the point where the FPR and FNR (1-TPR) are equal.
    # This can be approximated by finding the point where FPR and (1-TPR) are closest.
    differences = np.abs(fpr - (1 - tpr))
    min_difference_index = np.argmin(differences)
    eer = (fpr[min_difference_index] + (1 - tpr[min_difference_index])) / 2
    print(model, ": ", eer, " | ", "FPR: ", fpr[min_difference_index], " | ", "TPR: ", tpr[min_difference_index], " | ", "Precision: ", precision[min_difference_index])
# %%
# Compute EER with interpolation
for model in models:

    tpr = np.array(selfaware_data[model]["recall"])
    fpr = np.array(selfaware_data[model]["FPR"])
    precision = np.array(selfaware_data[model]["precision"])

    # Interpolate the TPR and FPR data for a finer resolution
    fpr_min, fpr_max = np.min(fpr), np.max(fpr)
    fpr_interp = np.linspace(fpr_min, fpr_max, 1000)  # Generating points within the range of provided FPR

    tpr_function = interp1d(fpr, tpr, kind='linear')  # Creating a linear interpolation function for TPR
    tpr_interp = tpr_function(fpr_interp)  # Interpolating TPR based on the interpolated FPR

    # Interpolate the precision and recall data for a finer resolution
    recall_min, recall_max = np.min(tpr), np.max(tpr)
    recall_interp = np.linspace(recall_min, recall_max, 1000)  # Generating points within the range of provided recall

    precision_function = interp1d(tpr, precision, kind='linear')  # Creating a linear interpolation function for precision
    precision_interp = precision_function(recall_interp)  # Interpolating precision based on the interpolated recall

    # Compute the differences between FPR and (1 - TPR) for the interpolated data
    differences_interp = np.abs(fpr_interp - (1 - tpr_interp))

    # Find the index of the minimum difference in the interpolated data
    min_difference_index_interp = np.argmin(differences_interp)

    # The EER is approximately the average of the FPR and TPR at this index in the interpolated data
    eer_interp = (fpr_interp[min_difference_index_interp] + (1 - tpr_interp[min_difference_index_interp])) / 2

    # eer_interp, min_difference_index_interp, fpr_interp[min_difference_index_interp], tpr_interp[min_difference_index_interp]

    print(model, ": ", "EER: ", eer_interp, " | FPR: ", fpr_interp[min_difference_index_interp], " | TPR: ", tpr_interp[min_difference_index_interp], " | Precision: ", precision_interp[min_difference_index_interp],
          "| F1: ", 2 * (precision_interp[min_difference_index_interp] * tpr_interp[min_difference_index_interp]) / (precision_interp[min_difference_index_interp] + tpr_interp[min_difference_index_interp]))

# %%
