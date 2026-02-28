# I-Predict-Therefore-I-Am

> Is Next Token Prediction Enough to Learn Human-Interpretable Concepts from Data?

This repository contains code for verifying two theoretical approximations:

1. The **product of weights from linear probing** and the **vector offset** computed from
   counterfactual pairs should approximate **1** in theory.

2. The **counterfactual pairs** used in our experiments can be downloaded from the
   [linear_rep_geometry](https://github.com/KihoPark/linear_rep_geometry) repository.

## 👨‍💻 Experiments

- The main code demonstrates how to verify the product-of-weights and vector-offset
  relationship through linear probing on saved activations.
- For experiments related to **structured sparse autoencoders (SAEs)**, please refer
  to our other recent work, **Concept Component Analysis (ConCA)**:
  https://github.com/yuhangliu01/ConCA

## 📥 Data and Dependencies

- Counterfactual pairs: download from
  https://github.com/KihoPark/linear_rep_geometry

## 📚 Citations

If you find this work helpful in your research, please cite the following papers:

```
@inproceedings{
  liu2026i,
  title={I Predict Therefore I Am: Is Next Token Prediction Enough to Learn Human-Interpretable Concepts from Data?},
  author={Yuhang Liu and Dong Gong and Yichao Cai and Erdun Gao and Zhen Zhang and Biwei Huang and Mingming Gong and Anton van den Hengel and Javen Qinfeng Shi},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=vVYD74U5KE}
}

@article{liu2026concept,
  title={Concept Component Analysis: A Principled Approach for Concept Extraction in LLMs},
  author={Liu, Yuhang and Gao, Erdun and Gong, Dong and Hengel, Anton van den and Shi, Javen Qinfeng},
  journal={arXiv preprint arXiv:2601.20420},
  year={2026}
}
```

---

Thank you for exploring this repository! Feel free to open issues or contribute.