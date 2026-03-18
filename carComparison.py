import pandas as pd
import random

def card_similarity(c1, c2, alpha_year=0.05, alpha_cond=0.5):
    dy = abs(c1["year"] - c2["year"])
    dc = abs(c1["condition"] - c2["condition"])
    # Larger exponent penalty → less influence when very different
    return 1.0 / (1.0 + alpha_year * dy + alpha_cond * dc)

def estimate_price(card_index, cards, adj):
    neighbors = adj[card_index]
    known = [(j, cards[j]["price"]) for j in neighbors if cards[j]["price"] is not None]
    if not known:
        return None  # no neighbor prices to anchor from

    weights = []
    for j, price in known:
        w = card_similarity(cards[card_index], cards[j])
        weights.append((w, price))

    total_w = sum(w for w, _ in weights)
    if total_w == 0:
        return None

    return sum(w * p for w, p in weights) / total_w