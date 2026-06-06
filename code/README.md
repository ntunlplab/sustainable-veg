## Workflow Pipeline


## Getting Started

Follow the steps below to set up and run the project pipeline.

### Step 1: Data Preparation
Download the required dataset and place all files into the `data/` directory at the project root.
```bash
mkdir -p data/
# Move your downloaded data files (e.g., reports/, sasb/) into the data/ folder. 
```

### Step 2: Build Database and Indices
```bash
# Option A: Build the primary database
python code/build_database.py

# Option B: Build the search index
python code/build_index.py
```

### Step 3: Run Approaches and Generate Predictions
```bash
# Run specific approaches (replace * with the actual script name, e.g., run_bm25.py)
python code/run_*.py
```

### Step 4: Evaluation and Scoring
```bash
python code/score.py
```

## Directory Structure
```text
├── data/                  # Place downloaded dataset here
├── code/
│   ├── build_database.py  # Builds the initial database
│   ├── build_index.py     # Constructs FAISS/BM25 indices
│   ├── run_*.py           # Inference/Retrieval scripts for different approaches
│   └── score.py           # Evaluation and scoring script
├── results/               # Generated prediction outputs
└── README.md
```

## TODO
1. 先照 .env.example 的格式建一份 .env 填上 api_key
2. run_direct.py 和 run_semantic.py 可能還有一些路徑錯誤，盡量改程式內部邏輯不要動到 config.yaml 定義的路徑
3. 檢查有沒有殘留 train/test 的判斷
4. 有些常用函數我搬到 utils.py 了，如果程式裡面有重複定義可以拿掉