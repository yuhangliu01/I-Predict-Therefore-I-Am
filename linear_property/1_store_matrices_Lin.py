import torch
import numpy as np
#import transformers
from tqdm import tqdm
import linear_rep_geometry as lrg
from sklearn.linear_model import RidgeClassifier, LogisticRegression


device = torch.device("cuda:0")

### load unembdding vectors ###
mm = lrg.model.name_or_path[-7:-1]

#gamma = lrg.model.lm_head.weight.detach().to(device) #Lamma

gamma = lrg.model.gpt_neox.embed_in.weight.detach().to(device) #pythia
W, d = gamma.shape


# gamma_bar = torch.mean(gamma, dim = 0)
# centered_gamma = gamma - gamma_bar

# ### compute Cov(gamma) and tranform gamma to g ###
# Cov_gamma = centered_gamma.T @ centered_gamma / W
# eigenvalues, eigenvectors = torch.linalg.eigh(Cov_gamma)
# inv_sqrt_Cov_gamma = eigenvectors @ torch.diag(1/torch.sqrt(eigenvalues)) @ eigenvectors.T
# sqrt_Cov_gamma = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
# g = gamma @ inv_sqrt_Cov_gamma

### compute concept directions ###
filenames = ['word_pairs/[verb - 3pSg].txt',
             'word_pairs/[verb - Ving].txt',
             'word_pairs/[verb - Ved].txt',
             'word_pairs/[Ving - 3pSg].txt',
             'word_pairs/[Ving - Ved].txt',
             'word_pairs/[3pSg - Ved].txt',
             'word_pairs/[verb - V + able].txt',
             'word_pairs/[verb - V + er].txt',
             'word_pairs/[verb - V + tion].txt',
             'word_pairs/[verb - V + ment].txt',
             'word_pairs/[adj - un + adj].txt',
             'word_pairs/[adj - adj + ly].txt',
             'word_pairs/[small - big].txt',
             'word_pairs/[thing - color].txt',
             'word_pairs/[thing - part].txt',
             'word_pairs/[country - capital].txt',
             'word_pairs/[pronoun - possessive].txt',
             'word_pairs/[male - female].txt',
             'word_pairs/[lower - upper].txt',
             'word_pairs/[noun - plural].txt',
             'word_pairs/[adj - comparative].txt',
             'word_pairs/[adj - superlative].txt',
             'word_pairs/[frequent - infrequent].txt',
             'word_pairs/[English - French].txt',
             'word_pairs/[French - German].txt',
             'word_pairs/[French - Spanish].txt',
             'word_pairs/[German - Spanish].txt'
             ]

concept_names = []

for name in filenames: ##word_pairs/[verb - 3pSg].txt
    content = name.split("/")[1].split(".")[0][1:-1] # verb - 3pSg
    parts = content.split(" - ")  # 0: verb , 1: 3pSg
    concept_names.append(r'${} \Rightarrow {}$'.format(parts[0], parts[1])) #'$verb \\Rightarrow 3pSg$'


count = 0

# embedding = []
# for filename in filenames:
#     embedding = []

#     first, second = lrg.get_embedding_pairs(filename)
#     embedding.append(first)
#     embedding.append(second)
    
#     first_second = filename.split("/")[1].split(".")[0][1:-1] # verb - 3pSg
#     first_and_second = first_second.split(" - ")  # 0: verb , 1: 3pSg
#     file_names=r'/data/Yuhang/linear_rep-main/data/deepseek{}_{}{}.pt'.format(mm, first_and_second[0], first_and_second[1])
#     torch.save(embedding, file_names)

#     count += 1
    
    
count = 0

#7b
np_embeddings = np.zeros((27, d))
np_weights = np.zeros((d, 27))


#13b
# np_embeddings = np.zeros((27, 5120))
# np_weights = np.zeros((5120, 27))

#70b
# np_embeddings = np.zeros((27, d))
# np_weights = np.zeros((d, 27))
for seed in range(10):
    
    np_embeddings = np.zeros((27, d))
    np_weights = np.zeros((d, 27))

    torch.manual_seed(seed)
    np.random.seed(seed)
    count = 0
    for filename in filenames:
        
        
        first_second = filename.split("/")[1].split(".")[0][1:-1] # verb - 3pSg
        first_and_second = first_second.split(" - ")  # 0: verb , 1: 3pSg
        file_names=r'/data/Yuhang/linear_rep-main/data/pythia{}_{}{}.pt'.format(mm, first_and_second[0], first_and_second[1])
        #file_names=r'/data/Yuhang/linear_rep-main/data/{}_{}{}.pt'.format(mm, first_and_second[0], first_and_second[1])#llema

        embedding = torch.load(file_names)
        
        N = len(embedding)
        half = N // 2  # 
        
        diff_data = embedding[0].cpu().detach().numpy() - embedding[1].cpu().detach().numpy()
        mean_diff_data = np.mean(diff_data, axis = 0, keepdims=True) #[:half]
        #mean_diff_data = np.mean(diff_data[:half], axis = 0, keepdims=True) #[:half]

        mean_diff_data = mean_diff_data / np.linalg.norm(mean_diff_data)
        
        np_embeddings[count,:] = mean_diff_data
        
        label = np.ones(len(embedding[0]) + len(embedding[1]))
        label[len(embedding[0]):] = 0
        
        np_embedding = np.concatenate( (embedding[0].cpu().detach().numpy(), embedding[1].cpu().detach().numpy()), 0 )
        
        #perm_label = np.random.permutation(label)   # null baseline

        clf = LogisticRegression(max_iter=500,C=0.0001).fit(np_embedding, label) # for most case C=0.0001, C=0.01 for deepseek 7b 14b
        #clf = LogisticRegression(max_iter=500,C=0.0001).fit(np_embedding[half:], label[half:]) # for most case C=0.0001, C=0.01 for deepseek 7b 14b

        perf = clf.score(np_embedding, label)
        print('Loss: {:.3f}'.format(perf))
        weights = clf.coef_/ np.linalg.norm(clf.coef_)
        
        np_weights[:,count]=weights

        count += 1
        #file_weights=r'/data/Yuhang/linear_rep-main/data/deepseek{}_{}{}_weights.pt'.format(mm, first_and_second[0], first_and_second[1])

    torch.save(np_weights, f'/data/Yuhang/linear_rep-main/data_new/holdout_pythia-1b_np_weights_seed{seed}.pt')
    torch.save(np_embeddings, f'/data/Yuhang/linear_rep-main/data_new/holdout_pythia-1b_np_embeddings_seed{seed}.pt')

    
    
    

