# Taxonomy level analysis of the different becteria

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd


PATH = r"C:\Users\batst\OneDrive\المستندات\CRC_Healthy+_Merged"

# Check the bacterias taxonomy level
def get_tax_level(tax_string):
    if "t__" in tax_string: return "Strain"      
    if "s__" in tax_string: return "Species"     
    if "g__" in tax_string: return "Genus"       
    if "f__" in tax_string: return "Family"      
    if "o__" in tax_string: return "Order"       
    if "c__" in tax_string: return "Class"       
    if "p__" in tax_string: return "Phylum"     
    if "k__" in tax_string: return "Kingdom"    
    return "Unknown"

# Count the bacterias presence for each taxonomy level
def taxonomy_analysis(df_abundances):
    taxa_cols = [col for col in df_abundances.columns if col != "sample_id"]

    levels = [get_tax_level(tax) for tax in taxa_cols]
    level_order = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species", "Strain"]

    level_counts = pd.Series(levels).value_counts().reindex(level_order).fillna(0)

    # Build graph
    plt.figure(figsize=(10, 6))
    sns.barplot(x=level_counts.index, y=level_counts.values, palette="viridis")
    plt.title(f"Bacteria Count per Taxonomic Level")
    plt.xlabel("Taxonomic Levels")
    plt.ylabel("Number of Bacteria")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save graph as image
    output_image = f"Taxonomic_Levels.png"
    plt.savefig(output_image)
    plt.show()


def main():
    file_paths = get_paths(PATH)

    # For each file analyse taxonomy level count
    for file in file_paths:
        print(f"Loading data...")
        df_abundances = pd.read_csv(file)
        taxonomy_analysis(df_abundances)

# Get abundances count files for each study
def get_paths(dirpath):
    directory = Path(dirpath)
    file_paths = [str(file) for file in directory.rglob("*abundances.csv")]
    print(file_paths)
    return file_paths

if __name__ == "__main__":
    main()
