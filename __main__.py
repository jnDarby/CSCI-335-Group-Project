import os
import joblib
import pandas as pd
from treeGeneration import build_similarity_tree, edges_to_adj

CACHE_FILE = "car_graph_cache.joblib"

def main():
    if os.path.exists(CACHE_FILE):
        print("Loading cached graph...")
        saved = joblib.load(CACHE_FILE)
        cards = saved["cards"]
        edges = saved["edges"]
        adj = saved["adj"]
    else:
        print("Building graph from scratch...")
        data = pd.read_csv("parsed_data.csv")

        required_cols = [
            "price", "year", "manufacturer", "model",
            "condition", "odometer"
        ]
        data = data.dropna(subset=required_cols)

        data = data.sample(n=2000, random_state=42)  # optional for speed
        cards = data.to_dict(orient="records")

        edges = build_similarity_tree(cards)
        adj = edges_to_adj(len(cards), edges)

        joblib.dump({
            "cards": cards,
            "edges": edges,
            "adj": adj
        }, CACHE_FILE)

        print("Graph saved to cache.")

    print("Cards:", len(cards))
    print("Edges:", len(edges))
    print("Neighbors of 0:", adj[0])

if __name__ == "__main__":
    main()