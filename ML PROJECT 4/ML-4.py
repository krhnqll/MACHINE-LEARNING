import numpy as np
import pandas as pd

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def gini_impurity(groups, classes):
    total_instances = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = sum((np.sum(group[:, -1] == c) / size) ** 2 for c in classes)
        gini += (1.0 - score) * (size / total_instances)
    return gini

def split_dataset(dataset, feature, threshold):
    left = dataset[dataset[:, feature] < threshold]
    right = dataset[dataset[:, feature] >= threshold]
    return left, right

def get_best_split(dataset):
    class_values = np.unique(dataset[:, -1])
    best_feature, best_threshold, best_score, best_groups = None, None, float('inf'), None

    for feature in range(dataset.shape[1] - 1):
        for threshold in np.unique(dataset[:, feature]):
            groups = split_dataset(dataset, feature, threshold)
            gini = gini_impurity(groups, class_values)
            if gini < best_score:
                best_feature, best_threshold, best_score, best_groups = feature, threshold, gini, groups

    return {'feature': best_feature, 'threshold': best_threshold, 'groups': best_groups}

def build_tree(dataset, max_depth, min_size, depth=0):
    X, y = dataset[:, :-1], dataset[:, -1]
    
    if len(np.unique(y)) == 1 or depth >= max_depth or len(dataset) <= min_size:
        leaf_value = np.bincount(y.astype(int)).argmax()
        return TreeNode(value=leaf_value)

    split = get_best_split(dataset)
    if not split['groups'] or len(split['groups'][0]) == 0 or len(split['groups'][1]) == 0:
        leaf_value = np.bincount(y.astype(int)).argmax()
        return TreeNode(value=leaf_value)

    left = build_tree(split['groups'][0], max_depth, min_size, depth + 1)
    right = build_tree(split['groups'][1], max_depth, min_size, depth + 1)

    return TreeNode(feature=split['feature'], threshold=split['threshold'], left=left, right=right)

def predict(tree, row):
    if tree.value is not None:
        return tree.value
    if row[tree.feature] < tree.threshold:
        return predict(tree.left, row)
    return predict(tree.right, row)

def load_data(train_path, test_path):
    train_data = pd.read_excel(train_path)
    test_data = pd.read_excel(test_path)
    return train_data.values, test_data

def main():
    train_path = 'trainDATA.xlsx'
    test_path = 'testDATA.xlsx'

    train_array, test_data = load_data(train_path, test_path)
    max_depth, min_size = 5, 10

    decision_tree = build_tree(train_array, max_depth, min_size)

    test_array = test_data.values
    predictions = [predict(decision_tree, row) for row in test_array]

    test_data['Car Acceptibility'] = predictions
    test_data.to_excel('test_predictions.xlsx', index=False)

if __name__ == "__main__":
    main()
