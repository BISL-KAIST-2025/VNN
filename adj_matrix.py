import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm


def df2adj(df):
    # Get unique values from both columns
    unique_source = sorted(df.iloc[:, 0].unique())
    unique_target = sorted(df.iloc[:, 1].unique())

    # Create a mapping from value to index for both columns
    source_index = {v: i for i, v in enumerate(unique_source)}
    target_index = {v: i for i, v in enumerate(unique_target)}

    # Initialize the adjacency matrix with zeros
    adj = np.zeros((len(unique_source), len(unique_target)))

    # Iterate through the DataFrame and update the adjacency matrix
    for _, row in tqdm(df.iterrows(), total=len(df)):
        source_idx = source_index[row.iloc[0]]
        target_idx = target_index[row.iloc[1]]
        adj[source_idx, target_idx] = 1

    return adj


def source2target_generator(df, source, target):
    # Create a mapping from value to index for both columns
    source_index = {v: i for i, v in enumerate(source)}
    target_index = {v: i for i, v in enumerate(target)}

    # Initialize the adjacency matrix with zeros
    adj = np.zeros((len(source), len(target)))

    # Iterate through the DataFrame and update the adjacency matrix
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row.iloc[0] in source_index.keys() and row.iloc[1] in target_index.keys():
            source_idx = source_index[row.iloc[0]]
            target_idx = target_index[row.iloc[1]]
            adj[source_idx, target_idx] = 1

    return adj


def main(data_path):
    # Load data
    gp2mf_df = pd.read_csv(data_path+'/gp2mf_df.csv')
    mf2bp_df = pd.read_csv(data_path+'/mf2bp_df.csv')
    
    # Generate adjacency matrices
    gp2mf_adj = df2adj(gp2mf_df)
    mf2bp_adj = df2adj(mf2bp_df)
    
    # Extract unique identifiers
    unique_gp = sorted(gp2mf_df['geneproductid'].unique())
    unique_mf = sorted(mf2bp_df['molecularfunctionid'].unique())
    unique_bp = sorted(mf2bp_df['biologicalprocessid'].unique())
    
    # Generate adjacency matrices for source to target relationships
    gp2mf_adj = source2target_generator(gp2mf_df, unique_gp, unique_mf)
    mf2bp_adj = source2target_generator(mf2bp_df, unique_mf, unique_bp)
    
    # Save adjacency matrices
    np.save(data_path+'/gp2mf_adj.npy', gp2mf_adj)
    np.save(data_path+'/mf2bp_adj.npy', mf2bp_adj)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct the biological adjacency matrix.")
    parser.add_argument("--data_path", default='./data', help="Path to the input files.")
    args = parser.parse_args()
    main(args.data_path)
