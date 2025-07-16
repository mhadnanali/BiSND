# BiSND: Binary Classification Social Network Dataset

**BiSND** is a benchmark dataset for graph machine learning, derived from real-world social network (Twitter) data.  
It contains node features, binary node labels, and three different graph structures (no edges, directed, undirected) to support a wide range of machine learning and network science experiments.

---

## Dataset Variants

This repository includes **three versions** of the BiSND dataset:

- [`no_edges/`](./no_edges/):  
  All nodes are disconnected (no edges). Useful for feature-only benchmarks and ablation studies.

- [`directed/`](./directed/):  
  Edges are **directed** (A → B if user A mentions B on Twitter).  
  Suitable for tasks needing edge directionality and real-world communication flow.

- [`undirected/`](./undirected/):  
  Edges are **undirected** (A—B if A and B mention each other).  
  Use for methods assuming undirected graphs, e.g., classic GNNs.

Each folder contains:
- `edges.csv` — List of graph edges (empty in `no_edges/`)
- `features.csv` — Node features (19 social/behavioral features)
- `labels.csv` — Node-level binary classification labels
- `train_mask.csv`, `val_mask.csv`, `test_mask.csv` — Standard splits for fair evaluation

---

## Node Features

Each node (Twitter user) is described by the following features:

- followers_count
- friends_count
- listed_count
- acc_created_at
- favourites_count
- verified
- Tweets
- followerPerDay
- statusPerDay
- favPerTweet
- friendsPerDay
- followFriends
- friendNlisted
- prot
- foPerTweet
- frPerTweet
- favPerFollow
- favPerFriend
- listPerDay

**Label:**  
- `label` in `labels.csv`:  
  - `1` = user still exists  
  - `0` = user deleted

---

## How to Load in PyTorch Geometric (PyG)

**Example (for `directed/` version):**
```python
import pandas as pd
import torch
from torch_geometric.data import Data

# Load edges
edges = pd.read_csv('directed/edges.csv')
edge_index = torch.tensor(edges.values.T, dtype=torch.long)

# Load features
features = pd.read_csv('directed/features.csv').sort_values('node_id')
x = torch.tensor(features.drop('node_id', axis=1).values, dtype=torch.float)

# Load labels
labels = pd.read_csv('directed/labels.csv').sort_values('node_id')
y = torch.tensor(labels['label'].values, dtype=torch.long)

# Optional: Load masks
train_ids = pd.read_csv('directed/train_mask.csv')['node_id'].values
val_ids = pd.read_csv('directed/val_mask.csv')['node_id'].values
test_ids = pd.read_csv('directed/test_mask.csv')['node_id'].values

train_mask = torch.zeros(x.size(0), dtype=torch.bool)
val_mask = torch.zeros(x.size(0), dtype=torch.bool)
test_mask = torch.zeros(x.size(0), dtype=torch.bool)
train_mask[train_ids] = True
val_mask[val_ids] = True
test_mask[test_ids] = True

# Create PyG Data object
data = Data(x=x, edge_index=edge_index, y=y,
            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
print(data)


```

# Properties of BiSND

**Benchmark dataset properties and statistics compared to BiSND, where ND: Node Degree, IN: Isolated Nodes, SL= self-Loops, UD=Undirected,  BF: Binary Features.**
| Dataset | Nodes | Feat. | Edges | Class | ND | ND Ratio | N/E | Non 0s | IN | SL | UD | BF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cora | 2708 | 1433 | 10556 | 7 | 3.90 | 17/1 | 25.7% | 1.27% | F | F | T | T |
| CiteSeer | 3327 | 3703 | 9104 | 6 | 2.74 | 13/0.7 | 36.5% | 0.85% | T | F | T | T |
| DBLP | 17,716 | 1639 | 105734 | 4 | 5.97 | 34/1 | 16.8% | 0.32% | F | F | T | T |
| PubMed | 19717 | 500 | 88648 | 3 | 4.50 | 29/1 | 22.2% | 10.0% | F | F | T | F |
| Actor | 7600 | 932 | 30019 | 5 | 3.95 | 19/0 | 25.3% | 0.58% | F | T | F | T |
| WikiCS | 11701 | 300 | 431726 | 10 | 36.90 | 229/0.4 | 2.7% | 99.99% | T | T | T | F |
| Am.Comp | 13752 | 767 | 491722 | 10 | 35.76 | 217/0.7 | 2.8% | 34.84% | T | F | T | T |
| Am.Photo | 7650 | 745 | 238162 | 8 | 31.13 | 168/0.95 | 3.2% | 34.74% | T | F | T | T |
| Co.CS | 18333 | 6805 | 163788 | 15 | 8.93 | 38/1 | 11.2% | 0.88% | F | F | T | T |
| Co.Phy | 34493 | 8415 | 495924 | 5 | 14.38 | 64/1.5 | 6.96% | 0.39% | F | F | T | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BiSND | 12788 | 19 | 430 | 2 | 0.01 | 0.67/0 | 0.30% | 80.13% | F | T | F | F | 


# Classification Accuracy

**Table: Classification accuracy of all algorithms where highest results of the table are bold.**
| Algorithm | Variant         | F1-score         | Precision        | Recall           | Jaccard Score    | Time      |
|-----------|-----------------|------------------|------------------|------------------|------------------|-----------|
| DT        | Entropy         | 67.42            | 63.21            | 82.39            | 55.69            | **0.19**  |
| KNN       | KNN             | 68.57            | 68.11        | 73.77            | 54.83            | 1.53      |
| RF        | Entropy         | 68.73        | 64.20            | 83.82        | 57.11        | 3.22      |
| XGB       | Gb Tree         | 68.49            | 65.25            | 78.28            | 55.25            | 53.09     |
| DNN       | MLP             | 66.83±0.46       | 64.25±0.96       | 71.18±3.87       | 50.91±1.42       | 90.49     |
| GNN       | Only Nodes      | 66.92±0.46       | 64.23±0.75       | 79.35±2.95       | 54.56±1.22       | 25    |
| GNN       | Un-directed     | 66.47±0.68       | 63.40±0.93       | 81.19±2.57       | 54.82±1.25       | 26        |
| GNN       | Directed        | 67.25±0.41   | 64.57±0.52   | 79.25±1.29   | 54.73±0.71   | 26        |
| GCL       | BGRL            | 66.26±0.70       | 63.65±0.15       | 76.12±0.56       | 53.11±0.64       | 1760      |
| GCL       | GRACE           | 67.39±0.37       | 63.27±0.15       | 77.75±0.31       | 53.20±0.13       | 856       |
| GCL       | DAENS₁D₁D       | 69.06±0.00       | 68.70±0.00       | 78.47±0.00       | 54.86±0.15       | 1376      |
| GCL       | DAENS₁D₂D       | 69.08±0.06       | 69.49±0.54       | 78.07±0.22       | 55.80±0.21       | 665   |
| GCL       | DAENS₂D₂D       | **70.15±0.25**   | **69.65±0.06**   | **86.27±0.31**   | **59.10±0.28**   | 715       |

# Article
The article is online as preprint at https://doi.org/10.48550/arXiv.2503.02397

## How to Cite

If you use BiSND in your research, please cite:

```bibtex
@misc{Ali2025BiSND,
      title={A Binary Classification Social Network Dataset for Graph Machine Learning}, 
      author={Adnan Ali and Jinglong Li},
      year={2025},
      eprint={2503.02397},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.02397}, 
}

