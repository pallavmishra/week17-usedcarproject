# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""
import argparse
from pathlib import Path
import mlflow
import mlflow.sklearn
import os
import json

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="used_cars_price_prediction_model", help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')
    return args

def main(args):
    '''Loads the best-trained model and registers it'''
    print("Registering ", args.model_name)
    
    # Step 1: Load the model from the specified path
    model = mlflow.sklearn.load_model(args.model_path)
    
    # Step 2: Log the loaded model in MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_price_regressor",
        registered_model_name=args.model_name
    )
    
    # Step 3: Register the logged model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/random_forest_price_regressor"
    registered_model = mlflow.register_model(model_uri, args.model_name)
    
    # Step 4: Write model registration details to JSON
    if args.model_info_output_path:
        model_info = {
            "model_name": args.model_name,
            "model_version": registered_model.version,
            "model_uri": model_uri
        }
        
        os.makedirs(args.model_info_output_path, exist_ok=True)
        output_path = os.path.join(args.model_info_output_path, "model_info.json")
        
        with open(output_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        
        print(f"Model info saved to: {output_path}")
    
    print(f"Model registered successfully as {args.model_name} version {registered_model.version}")

if __name__ == "__main__":
    mlflow.start_run()
    
    # Parse Arguments
    args = parse_args()
    
    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}"
    ]
    
    for line in lines:
        print(line)
    
    main(args)
    mlflow.end_run()
