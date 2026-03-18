import pandas as pd
import random
from dataParser import parse_data
from treeGeneration import random_tree_simple, edges_to_adj
from carComparison import card_similarity, estimate_price

def main():
    data = pd.read_csv('parsed_data.csv')
    n = len(data)
    edges = random_tree_simple(n)

if __name__ == "__main__":
    main()
    