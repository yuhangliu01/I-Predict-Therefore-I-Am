import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ----------------------------
# 配置
# ----------------------------
device = torch.device("cuda:0")
num_seeds = 10
model_name = "pythia-1b"
concept_num = 27  # number of concepts

# ----------------------------
# load concept names
# ----------------------------
concept_names = []
with open('matrices/concept_names.txt', 'r') as f:
    for index, line in enumerate(f.readlines()):
        concept_names.append(f"{line.strip()} ({index+1})")

concept_names_c = []
with open('matrices/concept_names_c.txt', 'r') as f:
    for index, line in enumerate(f.readlines()):
        concept_names_c.append(f"{line.strip()} ({index+1})")

# ----------------------------
# load seed data
# ----------------------------
g_stack = []
fro_list = []

for seed in range(num_seeds):
    weights_path = f'/data/Yuhang/linear_rep-main/data/holdout_{model_name}_np_weights_seed{seed}.pt'
    concepts_path = f'/data/Yuhang/linear_rep-main/data/holdout_{model_name}_np_embeddings_seed{seed}.pt'
    
    weights = torch.from_numpy(torch.load(weights_path)).to(device)  # (d, 27)
    concepts = torch.from_numpy(torch.load(concepts_path)).to(device) # (27, d)

    # inner product matrix = A_s W_s  shape (27,27)
    g_cos = concepts @ weights
    g_stack.append(g_cos.unsqueeze(0))

        # ---- Compute projection onto concept subspace ----
    A = concepts.T      # shape = (2048, 27)
    W = weights.T       # shape = (27, 2048)

    # A_s W_s
    AsWs = A @ W        # shape (2048, 2048)

    # Compute projection P_S = A (A^T A)^(-1) A^T
    AtA = A.T @ A       # shape (27, 27)
    inv_AtA = torch.inverse(AtA)

    P_S = A @ inv_AtA @ A.T   # (2048, 2048)

    fro_norm = torch.norm(AsWs - P_S, p='fro').item()
   
    fro_list.append(fro_norm)

g_stack = torch.cat(g_stack, dim=0)  # (num_seeds, 27, 27)

g_mean = g_stack.mean(dim=0).cpu().numpy()
g_std  = g_stack.std(dim=0).cpu().numpy()

fro_mean = np.mean(fro_list)
fro_std  = np.std(fro_list)

print("Frobenius mean =", fro_mean)
print("Frobenius std  =", fro_std)

# ----------------------------
# heatmap + Frobenius
# ----------------------------

fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(g_mean, cmap='coolwarm', vmin=0, vmax=1)

xtick = list(range(concept_num))
ytick = list(range(concept_num))

ax.set_xticks(xtick)
ax.set_yticks(ytick)
ax.set_xticklabels(xtick, rotation=90, fontsize=8)
ax.set_yticklabels(ytick, fontsize=8)

# 在每个格子中显示 std
for i in range(concept_num):
    for j in range(concept_num):
        text = f"{g_mean[i,j]:.2f}\n±{g_std[i,j]:.2f}"
        ax.text(j, i, text, ha="center", va="center", color="black", fontsize=6)


ax.set_title(f"{model_name}: Mean of $A_s W_s$ over {num_seeds} seeds", fontsize=20)

# colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)

# ---- 在下方写上 Frobenius mean ± std ----
ax.set_xlabel(
    rf"$\|\mathbf{{A}}_s \times \mathbf{{W}}_s - \mathbf{{P}}_{{\mathcal{{S}}}}\|_2 = {fro_norm:.4f}$",
    fontsize=22,
    labelpad=20  # 控制离图多远
)
plt.tight_layout()
plt.savefig(
    f"/data/Yuhang/linear_rep-main/figures/{model_name}_mean_heatmap_with_fro.png",
    dpi=300,
    bbox_inches='tight'
)
plt.close(fig)
