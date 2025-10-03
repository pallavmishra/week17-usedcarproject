# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""
import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--input_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")
    args = parser.parse_args()
    return args

def main(args):
    '''Read, preprocess, split, and save datasets'''
    # Reading Data
    df = pd.read_csv(args.input_data)
    
    # Step 1: Perform label encoding for the Segment column
    label_encoder = LabelEncoder()
    df['Segment'] = label_encoder.fit_transform(df['Segment'])
    
    # Step 2: Split the dataset into training and testing sets
    train_df, test_df = train_test_split(
        df, 
        test_size=args.test_train_ratio, 
        random_state=42
    )
    
    # Step 3: Save the training and testing datasets as CSV files
    output_train = os.path.join(args.train_data, "train.csv")
    output_test = os.path.join(args.test_data, "test.csv")
    
    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)
    
    # Step 4: Log the number of rows in datasets
    mlflow.log_metric("train_rows", train_df.shape[0])
    mlflow.log_metric("test_rows", test_df.shape[0])
    
    print(f"Training data saved to: {output_train}")
    print(f"Testing data saved to: {output_test}")

if __name__ == "__main__":
    mlflow.start_run()
    
    # Parse Arguments
    args = parse_args()
    
    lines = [
        f"Raw data path: {args.input_data}",
        f"Train dataset output path: {args.train_data}",
        f"Test dataset path: {args.test_data}",
        f"Test-train ratio: {args.test_train_ratio}",
    ]
    
    for line in lines:
        print(line)
    
    main(args)
    mlflow.end_run()
