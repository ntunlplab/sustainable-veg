# IR-2025-Final Project, Retriever

This README provides instructions for preparing the environment and executing inference on training set and testing set.  

  
## Prepare the Environment  
1. Navigate to the `retriever/` directory:
   ```bash
   cd code/retriever/
   ```
   
2. Install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```  

## Prerequisite
Preprocess the `train.json` file to enable subsequent scoring and ground-truth bounding-box visualization.
```bash
python convert_train_data.py
```

After running, a `train.csv` file will be generated, along with a `gt` directory containing the ground-truth bounding box visualizations.


## Build Embedding Database
```bash
python build_database.py
```

After running the script, a `sasb_db` directory will be generated, containing FAISS embedding databases for each individual metric.


## Inference  
```bash
python main.py
```

After running, the prediction results will be saved to `results.csv`, and the bounding-box visualizations will be stored in the `marked_results` directory.


## Evaluation
To compute the F1-score with ground truth, run:
```bash
python ../../data/score.py train.csv results.csv
```