import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt

plt.switch_backend('Agg') # Comment this in when we're only writing to disk and never displaying

def print_sep(statement, label="", expand=False):
    print(f"{label}=================================")
    if expand and isinstance(statement, list):
        for i in statement:
            print("  ", i)
    else:
        print(statement)
    print(f"{label}=================================")

class BaseReport:
    def __init__(
        self,
        dataframe,
        categorical_features = [],
        continuous_features = [],
        skip_features = [],
    ):
        self.dataframe = dataframe
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.skip_features = skip_features
        self._calculate_features()
        self._calculate_data_quality_report()

    def _calculate_features(self):
        for col in self.dataframe.columns:
            accounted_for = col in self.categorical_features or col in self.continuous_features or col in self.skip_features
            if accounted_for:
                continue
            if str(self.dataframe.dtypes[col]) in ["object", "bool"]:
                self.categorical_features.append(col)
            else:
                self.continuous_features.append(col)

    def _calculate_data_quality_report(self):
        self.continuouse_quality_report = pd.DataFrame({
            # ID
            "name": self.continuous_features,
            # Data Integrity
            "count": self.dataframe[self.continuous_features].count(),
            "% missing": self.dataframe[self.continuous_features].isnull().sum(),
            "cardinality": self.dataframe[self.continuous_features].nunique(),
            # Numeric Indicators
            "min": self.dataframe[self.continuous_features].min(),
            "1 quartile": self.dataframe[self.continuous_features].quantile(0.25),
            "mean": self.dataframe[self.continuous_features].mean(),
            "median": [self.dataframe[f].median() for f in self.continuous_features],
            "3 quartile": self.dataframe[self.continuous_features].quantile(0.75),
            "max": self.dataframe[self.continuous_features].max(),
            "std": self.dataframe[self.continuous_features].std(),
        })
        self.categorical_quality_report = pd.DataFrame({
            "name": self.categorical_features,
            "count": self.dataframe[self.categorical_features].count(),
            "% missing": self.dataframe[self.categorical_features].isnull().sum() / len(self.dataframe) * 100,
            "cardinality": self.dataframe[self.categorical_features].nunique(),
            "mode": [self.dataframe[f].mode()[0] for f in self.categorical_features],
        })

    def display(self):
        print_sep(self.dataframe.head(10), "HEAD")
        print_sep(self.dataframe.dtypes, "DTYPES")
        print_sep(self.dataframe.describe(), "DESCRIBE")
        print_sep(self.categorical_features, "categorical features".upper(), True)
        print_sep(self.continuous_features, "continuous features".upper(), True)
        print_sep(self.continuouse_quality_report, "CONTINUOUS REPORT")
        print_sep(self.categorical_quality_report, "CATEGORICAL REPORT")

    def write_splot(self, splot_path):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        full_path = os.path.join(dir_path, "../tmp", splot_path)
        if os.path.exists(full_path):
            shutil.rmtree(full_path)
        os.makedirs(full_path)
        for x_label in self.continuous_features:
            for y_label in self.continuous_features:
                _fig, ax = plt.subplots()

                ax.scatter(x=self.dataframe[x_label], y=self.dataframe[y_label])
                ax.set_ylabel(y_label)
                ax.set_xlabel(x_label)

                output_name = f"SPLOT-{x_label}-{y_label}.png"
                output_path = os.path.join(full_path, output_name)
                plt.savefig(output_path)
                plt.close()

class KDTreeLeaf:
    def __init__(self):
        self.id = None
        self.feature = None
        self.instance = None

        self.left = None
        self.right = None

    def __str__(self):
        return f"#{self.id} {self.feature} {self.value}"
    
    @property
    def value(self):
        return self.instance.get(self.feature)
class KDTree:
    def __init__(
        self,
        dataframe,
        features,
        identifying_feature="id"
    ):
        self.dataframe = dataframe
        self.features = features
        self.identifying_feature = identifying_feature
        self.root = self._build_node(self.dataframe, 0)

    def _build_node(self, data, feature_index):
        data_length = len(data.index)
        if data_length == 0:
            return None
        # Calculate feature and next feature
        feature = self.features[feature_index]
        next_feature_index = feature_index + 1
        if next_feature_index == len(self.features):
            next_feature_index = 0
        # To get the instance we care about we take the lowest value for the
        # current feature on the right side. This is because the right side can
        # actually include the median
        median = data[feature].median()
        right_set = data[data[feature] >= median].sort_values(feature)
        # Build the node
        node = KDTreeLeaf()
        node_instance_row = right_set.iloc[0]
        node.id = node_instance_row.get(self.identifying_feature)
        node.feature = feature
        node.instance = node_instance_row
        # recursively add the children
        if data_length > 1:
            right_set = right_set.iloc[1:, :]
            left_set = data[data[feature] < median]
            node.left = self._build_node(left_set, next_feature_index)
            node.right = self._build_node(right_set, next_feature_index)
        return node
    
    def _display_tree_text(self, node, depth):
        pad = "  " * depth
        print(pad + str(node.id) + "=====")
        if node.left is not None:
            self._display_tree_text(node.left, depth + 1)
        if node.right is not None:
            self._display_tree_text(node.right, depth + 1)

    def _get_depth_first_nodes(self, node):
        if node is None:
            return []
        return [node.id] + self._get_depth_first_nodes(node.left) + self._get_depth_first_nodes(node.right)

    def to_depth_first_array(self):
        return self._get_depth_first_nodes(self.root)
    
    def display_text(self):
        self._display_tree_text(self.root, 0)