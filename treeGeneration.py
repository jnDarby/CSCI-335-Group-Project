import pandas as pd
import random
import math

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def condition_to_number(condition):
    condition_map = {
        "new": 6,
        "like new": 5,
        "excellent": 4,
        "good": 3,
        "fair": 2,
        "salvage": 1
    }
    return condition_map.get(normalize_text(condition), 0)


def car_distance(c1, c2):
    score = 0.0

    year1 = safe_float(c1.get("year"))
    year2 = safe_float(c2.get("year"))
    score += 0.05 * abs(year1 - year2)

    odo1 = safe_float(c1.get("odometer"))
    odo2 = safe_float(c2.get("odometer"))
    score += abs(odo1 - odo2) / 50000.0

    cond1 = condition_to_number(c1.get("condition"))
    cond2 = condition_to_number(c2.get("condition"))
    score += 0.75 * abs(cond1 - cond2)

    if normalize_text(c1.get("manufacturer")) != normalize_text(c2.get("manufacturer")):
        score += 2.0

    if normalize_text(c1.get("model")) != normalize_text(c2.get("model")):
        score += 3.0

    if normalize_text(c1.get("fuel")) != normalize_text(c2.get("fuel")):
        score += 1.0

    if normalize_text(c1.get("transmission")) != normalize_text(c2.get("transmission")):
        score += 1.0

    if normalize_text(c1.get("drive")) != normalize_text(c2.get("drive")):
        score += 1.0

    if normalize_text(c1.get("type")) != normalize_text(c2.get("type")):
        score += 1.5

    if normalize_text(c1.get("title_status")) != normalize_text(c2.get("title_status")):
        score += 2.0

    return score


def car_similarity(c1, c2):
    
    d = car_distance(c1, c2)
    return 1.0 / (1.0 + d)


def edges_to_adj(n, edges):
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def build_similarity_graph(cards, k=3):
    n = len(cards)
    edge_set = set()

    for i in range(n):
        distances = []
        for j in range(n):
            if i == j:
                continue
            try:
                d = car_distance(cards[i], cards[j])
                if pd.isna(d):
                    continue
            except Exception:
                continue
            distances.append((d, j))

        distances.sort(key=lambda x: x[0])
        nearest = distances[:k]

        for _, j in nearest:
            edge = tuple(sorted((i, j)))
            edge_set.add(edge)

    return list(edge_set)


def build_similarity_tree(cards):
    n = len(cards)
    if n <= 1:
        return []

    visited = {0}
    edges = []

    next_report = 10

    while len(visited) < n:
        best_edge = None
        best_dist = math.inf

        for i in visited:
            for j in range(n):
                if j in visited:
                    continue

                d = car_distance(cards[i], cards[j])
                if d < best_dist:
                    best_dist = d
                    best_edge = (i, j)

        if best_edge is None:
            raise ValueError(
                "Could not find a valid edge to connect the remaining nodes. "
                "Check that cards is non-empty and each item is a valid dictionary."
            )

        edges.append(best_edge)
        visited.add(best_edge[1])

        print(f"{len(visited)}/{n} nodes connected)")

    print("100% complete")
    return edges

def random_tree_simple(n, seed=None):
    if n < 1:
        return []
    if seed is not None:
        random.seed(seed)

    edges = []
    for node in range(1, n):
        parent = random.randrange(0, node)
        edges.append((parent, node))
    return edges


def edges_to_adj(n, edges):
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def random_tree_prufer(n, seed=None):
    if n < 1:
        return []
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]

    rng = random.Random(seed)
    prufer = [rng.randrange(n) for _ in range(n - 2)]

    degree = [1] * n
    for x in prufer:
        degree[x] += 1

    edges = []
    for x in prufer:
        leaf = next(i for i in range(n) if degree[i] == 1)
        edges.append((leaf, x))
        degree[leaf] -= 1
        degree[x] -= 1

    last = [i for i in range(n) if degree[i] == 1]
    edges.append((last[0], last[1]))
    return edges
