# VNN
Visible Neural Network

## Requirements
### Python Version
- Python 3.8.5 is required.

### Required Packages
- Refer to the `requirements.txt` file for the list of required Python packages. Install them using:
  pip install -r requirements.txt

---

## Required Input Files
Input files required to run the code must be located in the `data` folder or specified using the `--data_path` argument. For example:
  python adj_matrix.py --data_path=input_path
  python main.py --data_path=input_path

### List of Input Files
1. `gp2mf_df.csv`
   - Contains relationships between gene products and molecular functions.

2. `mf2bp_df.csv`
   - Contains relationships between molecular functions and biological processes.
   
3. `embedding_x.npy`
   - Concatenated embeddings of herb embeddings and phenotype embeddings.

4. `embedding_y.npy`
   - Labels for the embeddings.
     - `0`: No relationship exists.
     - `1`: Relationship exists.

---

## How to Run the Analysis
### Step 1: Construct the Biological Adjacency Matrix
Run the following script to construct the adjacency matrix:
  python adj_matrix.py

### Step 2: Train and Evaluate the Model
Run the following command to train and evaluate the model:
  python main.py

### Additional Options
Use the `--help` flag to see all available arguments and options:
  python main.py --help
