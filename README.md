# Transaction Alert Prediction

This project addresses the Account Alert Prediction task by modeling financial transactions as a heterogeneous interaction graph.

## 📖 Model Overview

- Model the bank’s transaction data as a graph (accounts = nodes, transactions = edges).
- Compute node/edge behavioral features (amount patterns, channel/currency, time decay, structural statistics).
- Use random-walk path encoding to capture multi-hop transaction patterns.
- Apply an edge-aware Transformer GNN for alert prediction.
- Train the model using 5-fold cross-validation with class-weighted loss.

## 🛠️ Requirements

- python=3.9+
- networkx==3.2.1
- numpy==1.26.3
- pandas==2.3.3
- pyg-lib==0.4.0+pt27cu118
- scikit-learn==1.6.1
- torch==2.7.1+cu118
- torch-geometric==2.6.1
- torch_cluster==1.6.3+pt27cu118
- torch_scatter==2.1.2+pt27cu118
- torch_sparse==0.6.18+pt27cu118
- torch_spline_conv==1.2.2+pt27cu118
- torchaudio==2.7.1+cu118
- torchvision==0.22.1+cu118
- tqdm==4.67.1

## 📂 Project Structure

```bash
│── main.py                         # Unified entry point (run preprocess/train/inference)
│── README.md                       # Documentation and usage instructions
│
├── Cache/                          # Cached files: preprocessed graph, checkpoints
│
├── Dataset/                        # Raw dataset
│     ├── acct_transaction.csv      
│     ├── acct_alert.csv            
│     └── acct_predict.csv          
│
├── Model/                          
│     ├── model.py                  # GNN model definition
│     ├── train.py                  # Model training
│     ├── inference.py              # Inference + submission.csv generation
│     └── __init__.py               
│
├── Preprocess/                     # Preprocessing
│     ├── pre_proc.py               # Data cleaning, feature engineering, graph building
│     └── __init__.py
│
├── Results/                        # Final outputs
│     └── submission.csv            # Prediction submission file
│
└── Setup/                          # Environment / dependency configuration
      └── requirements.txt          # Python dependency list
```

## 🚀 Usage

1. Preprocessing
```bash
python main.py --mode preprocess
```
Output:
```bash
Cache/preprocessed_graph.pt
```
2. Training
```bash
python main.py --mode train
```
Output:
```bash
Cache/fold_1.pt
Cache/fold_2.pt
Cache/fold_3.pt
Cache/fold_4.pt
Cache/fold_5.pt
```
3. Inference
```bash
python main.py --mode infer
```
Output:
```bash
submission.csv
```

## 📊 Results
- F1 Score ~ 0.48