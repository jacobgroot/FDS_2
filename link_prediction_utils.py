"""
link_prediction_utils.py
Helper functions for steps that are repeated in the notebook.
Everything that runs only once stays visible in the notebook itself.
"""

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.model_selection import cross_val_score


# ── Communities ──────────────────────────────────────────────
# Used 4×: community plot (G), training features (G_feat),
# resolution search (G_feat), Kaggle features (G)
def community_map(graph, resolution):
    """Dict node -> community id, from greedy modularity on `graph`."""
    return {n: i for i, c in enumerate(greedy_modularity_communities(graph, resolution=resolution))
            for n in c}


# ── Features ─────────────────────────────────────────────────
# Used 2×: training pairs (on G_feat) and solutionInput pairs (on G)
def features(Gf, u, v, comm_of):
    """Feature vector of the pair (u, v), computed on graph Gf."""
    du, dv = Gf.degree(u), Gf.degree(v)
    return {
        "common_nb":   len(list(nx.common_neighbors(Gf, u, v))),
        "jaccard":     next(nx.jaccard_coefficient(Gf, [(u, v)]))[2],
        "adamic_adar": next(nx.adamic_adar_index(Gf, [(u, v)]))[2],
        "pref_attach": Gf.degree(u) * Gf.degree(v),
        "same_attr":   int(Gf.nodes[u]["attr"] == Gf.nodes[v]["attr"]),
        "same_comm":   int(comm_of[u] == comm_of[v]),
        # "max_deg":       max(du, dv),
    }


def feature_matrix(Gf, pairs, comm_of):
    """One row of features per (u, v) pair."""
    return pd.DataFrame([features(Gf, u, v, comm_of) for u, v in pairs])


# ── Plotting ─────────────────────────────────────────────────
# Used 3×: plain graph, coloured by attribute, coloured by community
def draw_network(G, pos, node_colors=None, node_size=5, figsize=(8, 8)):
    """Thin transparent edges + nodes (optionally coloured). Returns nothing; call plt.show() after extras."""
    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.3)
    if node_colors is None:
        nx.draw_networkx_nodes(G, pos, node_size=node_size)
    else:
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors)


# ── Model evaluation ─────────────────────────────────────────
# Used 4×: logistic regression, decision tree, DT vs RF, ensembles
def evaluate_model(name, model, X_train, y_train, X_test=None, y_test=None, cv=5):
    """
    Cross-validate `model` on the training data; if a test set is given,
    fit on X_train and score on X_test. Prints one line and returns the scores.
    """
    scores = cross_val_score(model, X_train, y_train, cv=cv)
    line = f"{name}: CV {scores.mean():.3f} ± {scores.std():.3f}"
    test_acc = None
    if X_test is not None:
        model.fit(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        line += f" | Test {test_acc:.3f}"
    print(line)
    return {"cv_mean": scores.mean(), "cv_std": scores.std(), "test_acc": test_acc}


# ── Grid search results ──────────────────────────────────────
# Used 2×: comparing the two searches, and declaring the final model
def search_summary(search, X_test, y_test):
    """Best params, CV mean ± std of the best combination, and test accuracy of a fitted GridSearchCV."""
    return {
        "best_params": search.best_params_,
        "cv_mean": search.best_score_,
        "cv_std": search.cv_results_["std_test_score"][search.best_index_],
        "test_acc": search.score(X_test, y_test),
    }
