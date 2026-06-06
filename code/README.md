## Workflow Pipeline


## Getting Started

Follow the steps below to set up and run the project pipeline.

### Step 1: Environment Setup
Before running any scripts, you need to configure your environment variables. Copy the example environment file and fill in your actual API key:
```bash
cp .env.example .env
# Open .env and add your api_key
```

### Step 2: Data Preparation
Download the required dataset and place all files into the `data/` directory at the project root.
```bash
mkdir -p data/
# Move your downloaded data files (e.g., reports/, sasb/) into the data/ folder. 
```

### Step 3: Build Database and Indices
```bash
# Option A: Build the primary database
python code/build_database.py

# Option B: Build the search index
python code/build_index.py
```

### Step 4: Run Approaches and Generate Predictions
```bash
# Run specific approaches (replace * with the actual script name, e.g., run_bm25.py)
python code/run_*.py
```

### Step 5: Evaluation and Scoring
```bash
python code/score.py
```
