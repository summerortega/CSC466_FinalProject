import pandas as pd
from .c45 import C45Tree

class RandomForest:
    #Only info gain needs to be used throughout the tests
    def __init__(self, num_attributes, num_datapoints, num_trees, splitting_metric='InfoGain',
                 splitting_threshold=0.05, random_state=None, max_depth=None, min_samples_leaf=1,
                 balanced_bootstrap=False):
        self.num_trees = num_trees
        self.num_datapoints = num_datapoints
        self.num_attributes = num_attributes
        self.splitting_metric = splitting_metric
        self.splitting_threshold = splitting_threshold
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.balanced_bootstrap = balanced_bootstrap
        self.forest = []

    def fit(self, x:pd.DataFrame, y:pd.Series, a:pd.Series):
        self.forest = []
        #For the n number of trees being created
        for _ in range(self.num_trees):
            seed = None if self.random_state is None else self.random_state + _
            #sample x datapoints from X and Y using replacement
            if 0 < self.num_datapoints <= 1:
                n = max(1, int(round(self.num_datapoints * len(x))))
            else:
                n = int(self.num_datapoints)
            if self.balanced_bootstrap:
                x_sample = self.balanced_sample(x, y, n, seed)
            else:
                x_sample = x.sample(n=n, replace=True, axis=0, random_state=seed)
            y_sample = y.loc[x_sample.index]
            #select y random attributes from the attribute set without replacement
            a_sample = a.sample(
                n=min(self.num_attributes, len(a)), replace=False, axis=0, random_state=seed
            ).reset_index(drop=True)
            #call the decision tree fit method with the sampled data and new attribute set
            new_tree = C45Tree(
                splitting_metric=self.splitting_metric,
                splitting_threshold=self.splitting_threshold,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            new_tree.fit(x_sample, y_sample, a_sample, self.splitting_threshold)
            #add the newly fitted tree to the forest
            self.forest.append(new_tree)

    def predict(self, x_test: pd.DataFrame) -> list[str]:
        predictions = [tree.predict(x_test) for tree in self.forest]
        results = []
        for i in range(len(x_test)):
            votes = [prediction[i] for prediction in predictions]
            counts = {}
            for v in votes:
                counts[v] = counts.get(v, 0) + 1
            max_count = max(counts.values())
            tied = sorted([label for label, c in counts.items() if c == max_count])
            results.append(tied[0])
        return results

    @staticmethod
    def balanced_sample(x: pd.DataFrame, y: pd.Series, n: int, seed: int | None) -> pd.DataFrame:
        classes = list(y.unique())
        base_n = max(1, n // len(classes))
        remainder = max(0, n - base_n * len(classes))
        samples: list[pd.DataFrame] = []

        for i, label in enumerate(classes):
            class_n = base_n + (1 if i < remainder else 0)
            class_index = y[y == label].index
            class_rows: pd.DataFrame = x.loc[class_index]
            sample_seed = None if seed is None else seed + i
            samples.append(class_rows.sample(n=class_n, replace=True, random_state=sample_seed))

        sampled = pd.concat(samples, axis=0)
        return sampled.sample(frac=1, random_state=seed)
