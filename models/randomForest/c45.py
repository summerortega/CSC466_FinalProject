from typing import Any

import pandas as pd
import numpy as np
import json
class C45Tree:
    def __init__(
            self, splitting_metric='Ratio', splitting_threshold=0.05,
            max_depth=None, min_samples_leaf=1
    ):
        self.splitting_metric = splitting_metric
        self.splitting_threshold = splitting_threshold
        self.max_depth = max_depth
        self.min_samples_leaf = max(1, int(min_samples_leaf))
        self.tree: dict[str, Any] = {}


    #x: Training Set
    #y: Training Ground Truth
    #a: Attribute Series of x
    #thresh: Algorithm Threshold
    def fit(self, x:pd.DataFrame, y:pd.Series, a:pd.Series, thresh:float, depth:int=0) -> dict[str, Any]:
        #Base Case 1: No Attributes Left to Split on
        if a.shape[0] == 0:
            decision = y.mode().values[0]
            p = len(y[y == decision]) / len(y)
            leaf = {"leaf":
                        {"decision": decision,
                         "p":p}
                  }
            self.tree = leaf
            return leaf
        #Base Case 2: y is homogenous
        elif y.value_counts().iloc[0] == y.shape[0]:
            decision = y.mode().values[0]
            p = len(y[y == decision]) / len(y)
            leaf = {"leaf":
                        {"decision": decision,
                         "p": p}
                    }
            self.tree = leaf
            return leaf
        #Base Case 3: Tree reached depth/sample limit
        elif ((self.max_depth is not None and depth >= self.max_depth)
              or y.shape[0] < 2 * self.min_samples_leaf):
            decision = y.mode().values[0]
            p = len(y[y == decision]) / len(y)
            leaf = {"leaf":
                        {"decision": decision,
                         "p": p}
                    }
            self.tree = leaf
            return leaf
        else:
            #select_split_att returns attribute name
            #and numeric split value (if it exists)
            att, split_val = select_split_att(
                x, y, a, thresh, mode=self.splitting_metric, min_samples_leaf=self.min_samples_leaf
            )
            #Case 1: No attribute returned; Return a leaf
            if not att:
                decision = y.mode().values[0]
                p = len(y[y == decision]) / len(y)
                leaf = {"leaf": {"decision": decision, "p": p}}
                self.tree = leaf
                return leaf
            #Case 2: Split Attribute Found
            else:
                edges: list[dict[str, Any]] = []
                att_tree: dict[str, Any] = {"node":
                                                {"var": att,
                                                 "edges": edges
                                                 }
                                            }
                #split_val being set to none
                #implies this is a categorical attribute
                if split_val is None:
                    # category
                    vals = x[att].unique()
                    #create a new tree for each possible value
                    for val in vals:
                        x_filtered = x[x[att] == val]
                        y_filtered =  y[x[att] == val]
                        if x_filtered.shape[0] != 0:
                            new_tree = self.fit(
                                x_filtered, y_filtered, a[a != att].reset_index(drop=True), thresh, depth + 1
                            )
                            edges.append({"edge": {"value": val } | new_tree })
                        else:
                            #create a ghost path and leaf if necessary
                            decision = y.mode().values[0]
                            p = len(y[y == decision]) / len(y)
                            ghost =  {"leaf":
                                        {"decision": decision,
                                         "p": p}
                                    }
                            edges.append({"edge": {"value": val} | ghost})
                else:
                    # numeric
                    x_filtered_lt = x[x[att] <= split_val]
                    y_filtered_lt = y[x[att] <= split_val]
                    x_filtered_gt = x[x[att] > split_val]
                    y_filtered_gt = y[x[att] > split_val]

                    #if all data is located on one side of the split
                    if x_filtered_gt.shape[0] == 0:
                        decision = y.mode().values[0]
                        p = len(y[y == decision]) / len(y)
                        return {"leaf":
                                    {"decision": decision,
                                     "p": p}
                                }

                    #else get left and right subtrees
                    #left child
                    new_tree = self.fit(x_filtered_lt, y_filtered_lt, a.reset_index(drop=True), thresh, depth + 1)
                    edges.append({"edge": {"value": split_val,
                                           "op": '<='} | new_tree})

                    #right child
                    new_tree = self.fit(x_filtered_gt, y_filtered_gt, a.reset_index(drop=True), thresh, depth + 1)
                    edges.append({"edge": {"value": split_val,
                                           "op": '>'} | new_tree})

                self.tree = att_tree
                return att_tree

    #return the prediction for a single observation
    def get_prediction(self, x:pd.Series) -> str:
        if "leaf" in self.tree:
            return self.tree["leaf"]["decision"]
        decision = None
        curr_node: dict[str, Any] = self.tree["node"]
        while not decision:
            curr_var = str(curr_node["var"])
            edge: dict[str, Any] | None = None
            first_edge: dict[str, Any] = curr_node["edges"][0]["edge"]

            if "op" in first_edge:
                for edge_wrapper in curr_node["edges"]:
                    candidate_edge = edge_wrapper["edge"]
                    if candidate_edge["op"] == "<=" and x[curr_var] <= candidate_edge["value"]:
                        edge = candidate_edge
                        break
                    elif candidate_edge["op"] == ">" and x[curr_var] > candidate_edge["value"]:
                        edge = candidate_edge
                        break
            else:
                categorical_edges =  {edge["edge"]["value"]: edge["edge"] for edge in curr_node["edges"]}
                edge = categorical_edges.get(x[curr_var])
            if edge is None:
                edge = curr_node["edges"][0]["edge"]
                while "node" in edge:
                    curr_node = edge["node"]
                    edge = curr_node["edges"][0]["edge"]
                return edge["leaf"]["decision"]
            if "node" in edge:
                curr_node = edge["node"]
            elif "leaf" in edge:
                decision = edge["leaf"]["decision"]
        return decision

    #get the predictions for an entire set of observations
    def predict(self, x_test:pd.DataFrame) -> list[str]:
        results = [self.get_prediction(x_test.iloc[i]) for i in range(len(x_test))]
        return results


    #creates/overwrited file, places tree into file as JSON
    #using JSON library
    def save_tree(self, save_file_path:str) -> None:
        with open(save_file_path, "w") as f:
            json.dump(self.tree, f, indent=2)


    #reads the JSON rendering of the tree from file, sets self.tree = result
    #using JSON library
    def read_tree(self, load_file_path:str) -> None:
        with open(load_file_path, "r") as f:
            self.tree = json.load(f)


#get splitting attribute from dataset
def select_split_att(
        x:pd.DataFrame, y:pd.Series, a:pd.Series, thresh:float, mode:str, min_samples_leaf:int=1
) -> tuple[str | None, float | None]:
    metric: list[float] = []
    numeric_splits: dict[str, float | None] = {}
    #for every unique attribute in Dataset
    #get info-gain or info-gain-ratio.
    for att_value in a:
        att = str(att_value)
        if x[att].dtypes == 'category':
            if x[att].value_counts().min() < min_samples_leaf:
                metric.append(-1)
                continue
            if mode == 'InfoGain':
                metric.append(info_gain(x, y, att))
            elif mode == 'Ratio':
                metric.append(info_gain_ratio(x, y, att))
        elif x[att].dtypes == 'float64':
            #numeric attributes choose metric in
            #find_best_split function.
            value, gain = find_best_split(x, y, att, mode, min_samples_leaf)
            numeric_splits[att] = value
            metric.append(gain)
    #choose best metric result
    #return attribute name and
    #numeric split value as needed
    if not metric:
        return None, None
    best = int(np.argmax(metric))
    if metric[best] >= thresh:
        best_att = str(a.iloc[best])
        if best_att in numeric_splits:
            return best_att, numeric_splits[best_att]
        else:
            return best_att, None
    else:
        return None, None


#get best numeric split value
#based on chosen metric
def find_best_split(
        x:pd.DataFrame, y:pd.Series, att:str, mode:str, min_samples_leaf:int=1
) -> tuple[float | None, float]:
    unique = np.unique(x[att])[1:]
    results: list[float] = []
    for val in unique:
        x_filtered_lt = x[x[att] <= val]
        y_filtered_lt = y[x[att] <= val]
        x_filtered_gt = x[x[att] > val]
        y_filtered_gt = y[x[att] > val]
        if x_filtered_lt.shape[0] < min_samples_leaf or x_filtered_gt.shape[0] < min_samples_leaf:
            continue

        entropy_binary_split = ((x_filtered_lt.shape[0]/x.shape[0] * entropy(y_filtered_lt)) +
                  (x_filtered_gt.shape[0]/x.shape[0] * entropy(y_filtered_gt)))
        if mode == "InfoGain":
            results.append(entropy(y) - entropy_binary_split)
        elif mode == "Ratio":
            gain = entropy(y) - entropy_binary_split
            results.append(info_gain_ratio_numeric(gain, x, y, att))
    if not results:
        return None, -1
    best = int(np.argmax(results))
    return float(unique[best]), results[best]


def info_gain(x:pd.DataFrame, y:pd.Series, att:str) -> float:
    unique = np.unique_counts(x[att])
    values = unique.values
    filtered_y_sets = [y[x[att] == val] for val in values]
    entropy_split = np.sum([y_prime.shape[0]/y.shape[0] * entropy(y_prime) for y_prime in filtered_y_sets])
    return float(entropy(y) - entropy_split)


def info_gain_ratio(x:pd.DataFrame, y:pd.Series, att:str) -> float:
    gain = info_gain(x, y, att)
    unique = np.unique_counts(x[att])
    values = unique.values
    filtered_y_shapes = [y[x[att] == val].shape[0] for val in values]
    y_shape = y.shape[0]
    den = [(filtered_y/y_shape) * (np.log2((filtered_y/y_shape))) for filtered_y in filtered_y_shapes]
    denom = -1 * sum(den)
    if denom == 0:
        return 0.0
    return gain / denom


#specially made info-gain-ratio function
#for use with the find_best_split function
def info_gain_ratio_numeric(gain:float, x:pd.DataFrame, y:pd.Series, att:str) -> float:
    unique = np.unique_counts(x[att])
    values = unique.values
    filtered_y_shapes = [y[x[att] == val].shape[0] for val in values]
    y_shape = y.shape[0]
    den = [(filtered_y / y_shape) * (np.log2((filtered_y / y_shape))) for filtered_y in filtered_y_shapes]
    denom = -1 * sum(den)
    if denom == 0:
        return 0.0
    return gain / denom


def entropy(y:pd.Series) -> float:
    if y.value_counts().iloc[0] == y.shape[0]:
        return 0
    unique = np.unique_counts(y)
    counts = unique.counts
    total = np.sum(counts)
    calculations = [(counts[i] / total) * np.log2((counts[i] / total)) for i in range(len(counts))]

    return -1 * np.sum(calculations)
