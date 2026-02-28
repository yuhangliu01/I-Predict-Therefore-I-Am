import transformers
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import json
from tqdm import tqdm

device = torch.device("cuda:0")
torch.set_num_threads(8)
sns.set_theme(
    context="paper",
    style="white",  # 'whitegrid', 'dark', 'darkgrid', ...
    palette="colorblind",
    font="sans-serif",  # 'serif'
    font_scale=1.75,  # 1.75, 2, ...
)

# MODEL_PATH = "/lts/yhliu/Pre-trained LLMs/models--meta-llama--Meta-Llama-3-70B" #13 70
# model_name = "DeepSeek-R1-Distill-Qwen-7B"  #"Meta-Llama-3-70B"
# a = f"meta-llama/{model_name}"

# tokenizer = transformers.AutoTokenizer.from_pretrained(f"deepseek-ai/{model_name}", cache_dir = '/lts/yhliu/Pre-trained LLMs',load_in_8bit=True)
# model = transformers.AutoModelForCausalLM.from_pretrained(f"deepseek-ai/{model_name}",  low_cpu_mem_usage=True, trust_remote_code=True, cache_dir = '/lts/yhliu/Pre-trained LLMs')
# #model.to(device)

#7b and 13b
# tokenizer = transformers.AutoTokenizer.from_pretrained(f"meta-llama/{model_name}", cache_dir = '/lts/yhliu/Pre-trained LLMs')
# model = transformers.AutoModelForCausalLM.from_pretrained(f"meta-llama/{model_name}", low_cpu_mem_usage=True, device_map="auto",cache_dir = '/lts/yhliu/Pre-trained LLMs')
# #model.to(device)
# 70b 
# tokenizer = transformers.AutoTokenizer.from_pretrained(f"meta-llama/{model_name}", cache_dir = '/lts/yhliu/Pre-trained LLMs',load_in_8bit=True)
# model = transformers.AutoModelForCausalLM.from_pretrained(f"meta-llama/{model_name}", low_cpu_mem_usage=True, cache_dir = '/lts/yhliu/Pre-trained LLMs', load_in_8bit=True)



#MODEL_PATH = "/lts/yhliu/Pre-trained LLMs/models--meta-llama--Llama-2-70b-hf" #13 70
# model_name = "Llama-2-7b-hf"
# #model_name = "Llama-2-13b-hf"
# #model_name = "Llama-2-70b-hf"
# tokenizer = transformers.LlamaTokenizer.from_pretrained(f"meta-llama/{model_name}", cache_dir = '/lts/yhliu/Pre-trained LLMs')
# model = transformers.LlamaForCausalLM.from_pretrained(f"meta-llama/{model_name}", low_cpu_mem_usage=True, device_map="auto",cache_dir = '/lts/yhliu/Pre-trained LLMs')
# #model = transformers.LlamaForCausalLM.from_pretrained(f"meta-llama/{model_name}", torch_dtype=torch.float16, device_map="auto",cache_dir = '/lts/yhliu/Pre-trained LLMs')




# MODEL_PATH = "/lts/yhliu/Pre-trained LLMs/models--EleutherAI--pythia-12b" # 
model_name = "pythia-1b"
model_name = "pythia-1.4b"
model_name = "pythia-2.8b"
model_name = "pythia-6.9b"
model_name = "pythia-12b"
# pythia-1b 
tokenizer = transformers.AutoTokenizer.from_pretrained(f"EleutherAI/{model_name}", cache_dir = '/lts/yhliu/Pre-trained LLMs')
model = transformers.AutoModelForCausalLM.from_pretrained(f"EleutherAI/{model_name}", low_cpu_mem_usage=True, device_map="auto",cache_dir = '/lts/yhliu/Pre-trained LLMs')
# print(type(model))

## get indices of counterfactual pairs
def get_counterfactual_pairs(filename):
    fullpath = '/data/Yuhang/linear_rep-main/' + filename
    with open(fullpath, 'r') as f:
        lines = f.readlines()
    words_pairs = [line.strip().split('\t') for line in lines if line.strip()]

    base_ind = []
    target_ind = []

    for i in range(len(words_pairs)):
        first = tokenizer.encode(words_pairs[i][0])
        second = tokenizer.encode(words_pairs[i][1])
        if len(first) == len(second) == 2 and first[1] != second[1]:
            base_ind.append(first[1])
            target_ind.append(second[1])
    base_name = [tokenizer.decode(i) for i in base_ind]
    target_name = [tokenizer.decode(i) for i in target_ind]

    return base_ind, target_ind, base_name, target_name



###
def get_embedding_pairs(filename):
    fullpath = '/data/Yuhang/linear_rep-main/' + filename
    with open(fullpath, 'r') as f:
        lines = f.readlines()
    words_pairs = [line.strip().split('\t') for line in lines if line.strip()]

    lambdas_0 = []
    lambdas_1 = []

    for i in range(len(words_pairs)):
        first =  words_pairs[i][0] 
        second =  words_pairs[i][1] 
        lambdas_0.append(get_embeddings(first))
        lambdas_1.append(get_embeddings(second))
    

    return torch.cat(lambdas_0), torch.cat(lambdas_1)

## get concept direction
def concept_direction(base_ind, target_ind, data):
    base_data = data[base_ind,]; target_data = data[target_ind,]

    diff_data = target_data - base_data
    mean_diff_data = torch.mean(diff_data, dim = 0)
    mean_diff_data = mean_diff_data / torch.norm(mean_diff_data)

    return mean_diff_data, diff_data

## get embeddings of each text
def get_embeddings(text_batch):
    tokenizer.pad_token = tokenizer.eos_token
    
    #7b and 13b
    #tokenized_output = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True).to(device)
    #70b
    #tokenized_output = tokenizer(text_batch, return_tensors="pt", padding=True, max_length=tokenizer.model_max_length, truncation=True).to(device)
    
    #pythia
    tokenized_output = tokenizer(text_batch, return_tensors="pt", padding=True, max_length=512, truncation=True).to(device)

    
    input_ids = tokenized_output["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states = True)
    hidden_states = outputs.hidden_states

    seq_lengths = tokenized_output.attention_mask.sum(dim=1).tolist()
    last_token_positions = [length - 1 for length in seq_lengths]
    text_embeddings = torch.stack([hidden_states[-1][i, pos, :] for i, pos in enumerate(last_token_positions)])

    return text_embeddings









## draw heatmap of the inner products
def draw_heatmaps(data_matrices, concept_labels, concept_labels_c, cmap = 'PiYG'):
    fig = plt.figure(figsize=(14, 8.5))
    gs = gridspec.GridSpec(2, 3, wspace=0.2)
    
    vmin = min([data.min() for data in data_matrices])
    vmax = max([data.max() for data in data_matrices])
    
    ticks = list(range(2, 27, 3))
    labels = [str(i+1) for i in ticks]
    
    ytick = list(range(27))
    xtick = list(range(27))
    ims = []

    ax_left = plt.subplot(gs[0:2, 0:2])
    im = ax_left.imshow(data_matrices[0], cmap=cmap,vmin=0, vmax=1)
    ims.append(im)
    # ax_left.set_xticks(ticks)
    # ax_left.set_xticklabels(labels)
    ax_left.set_xticks(xtick)
    ax_left.set_xticklabels(xtick, rotation=90)   

    
    ax_left.set_yticks(ytick)
    #ax_left.set_yticklabels(concept_labels)
    #ax_left.set_title(r'$M = \mathrm{Cov}(\gamma)^{-1}$')
    ax_left.set_title(r'$\mathbf{A}_s \times \mathbf{W}_s$ on Pythia-1b', fontsize=32)

    # ax_top_right = plt.subplot(gs[0, 2])
    # im = ax_top_right.imshow(data_matrices[1], cmap=cmap)
    # ims.append(im)
    # ax_top_right.set_xticks([])
    # ax_top_right.set_yticks([])
    # ax_top_right.set_title(r'$M = I_d$')

    # ax_bottom_right = plt.subplot(gs[1, 2])
    # im = ax_bottom_right.imshow(data_matrices[2], cmap=cmap)
    # ims.append(im)
    # ax_bottom_right.set_xticks([])
    # ax_bottom_right.set_yticks([])
    # ax_bottom_right.set_title(r'Random $M$')
    
    divider = make_axes_locatable(ax_left)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(ims[-1], cax=cax, orientation='vertical')
    

    plt.tight_layout()
    plt.subplots_adjust(left=0.15)
    plt.savefig(f"/data/Yuhang/linear_rep-main/F_holdout_pythia-1b_seed0.png", bbox_inches='tight')
    plt.show()

## draw heatmap of the inner products
def draw_heatmaps_F(data_matrices, concept_labels, concept_labels_c, fro_norm=None, cmap='PiYG'):
    fig = plt.figure(figsize=(14, 8.5))
    gs = gridspec.GridSpec(2, 3, wspace=0.2)
    
    vmin = min([data.min() for data in data_matrices])
    vmax = max([data.max() for data in data_matrices])
    
    ytick = list(range(8))
    xtick = list(range(8))
    ims = []

    # ----------------------------
    # Main heatmap panel
    # ----------------------------
    ax_left = plt.subplot(gs[0:2, 0:2])
    im = ax_left.imshow(data_matrices[0], cmap=cmap, vmin=0, vmax=1)
    ims.append(im)

    ax_left.set_xticks(xtick)
    ax_left.set_xticklabels(xtick, rotation=90)
    ax_left.set_yticks(ytick)
    ax_left.set_title(r'$\mathbf{A}_s \times \mathbf{W}_s$ on Pythia-1b', fontsize=32)

    # ----------------------------
    # Colorbar + Frobenius Norm
    # ----------------------------
    divider = make_axes_locatable(ax_left)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(ims[-1], cax=cax, orientation='vertical')

    # ---- ADD THIS: display scalar near colorbar ----
    ax_left.set_xlabel(
        rf"$\|\mathbf{{A}}_s \times \mathbf{{W}}_s - \mathbf{{P}}_{{\mathcal{{S}}}}\|_2 = {fro_norm:.4f}$",
        fontsize=22,
        labelpad=20  # 
    )

    # ----------------------------
    plt.tight_layout()
    plt.subplots_adjust(left=0.15)
    plt.savefig(f"/data/Yuhang/linear_rep-main/F_new_perm_Pythia-1b_seed0.png",
                bbox_inches='tight')
    plt.show()



####### Experiment 3: measurement #######
## get lambda pairs for diffrenet languages
def get_lambda_pairs(filename, num_eg = 20):
    lambdas_0 = []
    lambdas_1 = []

    count =0
    with open(filename, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            data = json.loads(line)
            if count >= num_eg:
                break

            text_0 = [s.strip(" " + data['word0']) for s in data['contexts0']]
            lambdas_0.append(get_embeddings(text_0))

            text_1 = [s.strip(" " + data['word1']) for s in data['contexts1']]
            lambdas_1.append(get_embeddings(text_1))
            
            count += 1

    return torch.cat(lambdas_0), torch.cat(lambdas_1)

## show histogram of lambda^T gamma_W
def hist_measurement(lambda_0, lambda_1, concept, concept_names,
                    base = "English", target = "French", alpha = 0.5):
    fig, axs = plt.subplots(7, 4, figsize=(16, 20))

    axs = axs.flatten()

    for i in range(concept.shape[0]):
        W0 = lambda_0 @ concept[i]
        W1 = lambda_1 @ concept[i]

        axs[i].hist(W0.cpu().numpy(), bins = 25, alpha=alpha, label=base, density=True)
        axs[i].hist(W1.cpu().numpy(), bins = 25, alpha=alpha, label=target,  density=True)
        axs[i].set_yticks([])
        axs[i].set_title(f'{concept_names[i]}')

    handles, labels = axs[0].get_legend_handles_labels()
    axs[concept.shape[0]].legend(handles, labels, loc='center')
    axs[concept.shape[0]].axis('off')

    plt.tight_layout()
    plt.savefig("figures/appendix_measurement_"+ base + "-" + target + ".pdf", bbox_inches='tight')
    plt.show()





