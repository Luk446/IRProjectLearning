import pandas as pd
import numpy as np
import ast
import sys

# EXAMPLE USAGE:
# python3 convert_to_npy.py ../controllers/supervisorGA_starter/data/genome_data_20251115-211540.csv 52 10./Best.npy

def save_genotype_npy(csv_file, generation, population, output_npy):
    # Load CSV
    df = pd.read_csv(csv_file)
    # Find the specific row
    match = df[(df['generation'] == int(generation)) & (df['population'] == int(population))]
    if match.empty:
        raise ValueError("No match found for generation={} population={}".format(generation, population))
    # Convert genotype string to array
    genotype_str = match.iloc[0]['genotype']
    genotype_list = ast.literal_eval(genotype_str)
    genotype_array = np.array(genotype_list, dtype=float)
    # Save as .npy
    np.save(output_npy, genotype_array)
    print(f"Saved {output_npy}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <csv_file> <generation> <population> <output_npy>")
        sys.exit(1)
    save_genotype_npy(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
