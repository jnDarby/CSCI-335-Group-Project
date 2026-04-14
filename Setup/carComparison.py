import pandas as pd
import random
from treeGeneration import *

def card_similarity(c1, c2, alpha_year=0.05, alpha_cond=0.5):
    dy = abs(float(c1["year"]) - float(c2["year"]))
    dc = abs(condition_to_number(c1["condition"]) - condition_to_number(c2["condition"]))
    return 1.0 / (1.0 + alpha_year * dy + alpha_cond * dc)


def estimate_price(card_index, cards, adj):
    neighbors = adj[card_index]
    known = [(j, cards[j]["price"]) for j in neighbors if cards[j]["price"] is not None]

    if not known:
        return None

    weights = []
    for j, price in known:
        w = card_similarity(cards[card_index], cards[j])
        weights.append((w, price))

    total_w = sum(w for w, _ in weights)
    if total_w == 0:
        return None

    return sum(w * p for w, p in weights) / total_w