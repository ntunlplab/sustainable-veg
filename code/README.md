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