import numpy as np
import pandas as pd
import os

def load_data(file_path):
    train_data = pd.read_excel(file_path, sheet_name='TRAINData')
    test_data = pd.read_excel(file_path, sheet_name='TESTData')

    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    return X_train, y_train, X_test, y_test, test_data

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.array([1 if i > 0 else 0 for i in y])  

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation_function(linear_output)

                # Update rule
                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._activation_function(linear_output)
        return y_predicted

    def _activation_function(self, x):
        return np.where(x >= 0, 1, 0)

def main():
    file_path = "hw3/DataForPerceptron.xlsx"
    X_train, y_train, X_test, y_test, test_data = load_data(file_path)

    perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)
    perceptron.fit(X_train, y_train)

    predictions = perceptron.predict(X_test)
    accuracy = np.mean(predictions == y_test) * 100
    print(f'Accuracy on test data: {accuracy:.2f}%')

    
    output_df = pd.DataFrame(X_test, columns=test_data.columns[:-1])
    output_df['Predicted'] = predictions
    output_file_path = "PredResults.xlsx"
    output_df.to_excel(output_file_path, index=False)
    print(f'Pred saved to {output_file_path}')

if __name__ == "__main__":
    main()
