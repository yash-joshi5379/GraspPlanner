import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from classification.NeuralNetwork import NeuralNetwork


class Classifier():
    """
    A machine learning classifier for grasp success prediction.

    This class provides functionality to train, test, and evaluate different
    classification models (Random Forest, Logistic Regression, Neural Network, or SVM)
    on grasp success datasets. It handles data preprocessing, model training with
    hyperparameter optimization, and model persistence.
    """

    def __init__(self, dataset, model_choice):
        """
        Initialize the Classifier.

        Args:
            dataset (str): Path to the CSV dataset file containing features and labels.
            model_choice (str): Type of model to use ('R' for Random Forest, 'L' for
                Logistic Regression, 'N' for Neural Network, or 'S' for SVM).
        """
        self.dataset = dataset
        self.model_choice = str(model_choice).upper()
        self.model = None
        self.best_params = None
        self.scaler = StandardScaler()

    def split_dataset(self):
        """
        Split the dataset into training and testing sets with feature scaling.

        Reads the CSV dataset, removes unnamed index columns, separates features
        and labels, splits into 70% training and 30% testing sets with stratification,
        and applies standard scaling to both sets (fitted on training data only to
        prevent data leakage).

        Sets instance attributes:
            X_train, X_test: Scaled feature arrays
            y_train, y_test: Label arrays
        """
        df = pd.read_csv(self.dataset)

        # Remove index column if it exists (Unnamed: 0)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        y = df['label'].values
        X = df.drop(['label'], axis=1).values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)

        # scale only on training set to avoid data leakage
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def get_model_and_params(self):
        """
        Return a base model and its hyperparameter grid based on the model choice.

        Returns:
            tuple: A tuple of (model, param_grid) where:
                - model: An instantiated classifier model
                - param_grid: A dictionary of hyperparameters to search over for GridSearchCV

        The model choice determines which classifier and parameters are used:
            'R': Random Forest with n_estimators and max_depth
            'L': Logistic Regression with C parameter
            'N': Neural Network with batch_size, epochs, and dropout
            'S' (default): SVM with C and gamma parameters
        """
        if self.model_choice == 'R':
            model = RandomForestClassifier(
                random_state=42, class_weight='balanced')
            params = {
                'n_estimators': np.arange(100, 301, 20),
                'max_depth': np.arange(5, 11, 1)
            }
        elif self.model_choice == 'L':
            model = LogisticRegression(penalty='l2', class_weight='balanced')
            params = {
                'C': [0.01, 0.1, 1, 10]
            }
        elif self.model_choice == 'N':
            model = NeuralNetwork()
            params = {
                'batch_size': [6, 8, 10, 12],
                'epochs': [40, 50, 60, 70],
                'dropout': [0.2, 0.3, 0.4]
            }
        else:
            model = SVC(probability=True, kernel='rbf', random_state=42)
            params = {
                'C': np.arange(0.1, 5.1, 0.1),
                'gamma': ['scale', 'auto']
            }
        return model, params

    def train_model(self):
        """
        Train the classifier using GridSearchCV for hyperparameter optimization.

        Performs 5-fold cross-validation over the parameter grid using the F1-weighted
        scoring metric. The best estimator and parameters found are stored in the
        instance attributes self.model and self.best_params. Prints the best parameters.
        """
        base_model, param_grid = self.get_model_and_params()

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='f1_weighted',
            cv=5,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(self.X_train, self.y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_

        print("Best params found:", self.best_params)

    def test_model(self):
        """
        Test the trained model on the test set and display performance statistics.

        Makes predictions on the test set and calls display_stats to print and
        visualize the classification metrics and confusion matrix.
        """
        y_pred = self.model.predict(self.X_test)
        self.display_stats(
            self.y_test,
            y_pred,
            self.dataset,
            self.model_choice)

    @staticmethod
    def display_stats(y_test, y_pred, dataset_path, model_choice):
        """
        Display and save classification performance metrics and confusion matrix.

        Prints accuracy and per-class precision, recall, and F1-score for both classes.
        Generates and saves a confusion matrix visualization as a PNG file in the
        'confusion_matrices' directory.

        Args:
            y_test (array-like): True labels from the test set.
            y_pred (array-like): Predicted labels from the model.
            dataset_path (str): Path to the dataset file, used for naming the output figure.
            model_choice (str): The model type identifier, used for naming the output figure.
        """
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

        # metrics by class (labels sorted as [0, 1])
        f1_neg, f1_pos = f1_score(y_test, y_pred, average=None, labels=[0, 1])
        precision_neg, precision_pos = precision_score(
            y_test, y_pred, average=None, labels=[0, 1])
        recall_neg, recall_pos = recall_score(
            y_test, y_pred, average=None, labels=[0, 1])

        print(
            f"Class 0: Precision = {
                precision_neg:.4f} | Recall = {
                recall_neg:.4f} | F1-score = {
                f1_neg:.4f}")
        print(
            f"Class 1: Precision = {
                precision_pos:.4f} | Recall = {
                recall_pos:.4f} | F1-score = {
                f1_pos:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
        cm_display.plot(cmap='Blues')
        for text in cm_display.text_.ravel():
            text.set_fontsize(32)
        os.makedirs("confusion_matrices", exist_ok=True)
        dataset_stem = os.path.splitext(os.path.basename(dataset_path))[0]
        fig_name = f"{dataset_stem}_Model_{str(model_choice).upper()}.png"
        plt.tight_layout()
        plt.savefig(os.path.join("confusion_matrices", fig_name))
        plt.close()

    def save_model(self):
        """
        Save the trained model and scaler to a pickle file.

        Saves the model, scaler, model choice, dataset path, and best parameters
        to a pickle file in the 'models' directory. The filename is generated based
        on the dataset name and model type.
        """

        # Organise all model data into a dictionary
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_choice': self.model_choice,
            'dataset': self.dataset,
            'best_params': self.best_params
        }

        # Get the file name for this dataset and model
        os.makedirs("models", exist_ok=True)
        dataset_stem = os.path.splitext(os.path.basename(self.dataset))[0]
        model_name = f"{dataset_stem}_model_{self.model_choice}.pkl"
        model_path = os.path.join("models", model_name)

        # Save the model data as a .pkl file
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_path}")

    @staticmethod
    def load_model(filepath):
        """
        Load a trained model and scaler from a pickle file.

        Args:
            filepath (str): Path to the pickle file containing the saved model data.

        Returns:
            Classifier: A Classifier instance with the loaded model, scaler, and parameters.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        classifier = Classifier(
            model_data['dataset'],
            model_data['model_choice'])
        classifier.model = model_data['model']
        classifier.scaler = model_data['scaler']
        classifier.best_params = model_data.get('best_params', None)
        return classifier

    def test_on_new_data(self, test_dataset):
        """
        Test the trained model on a new dataset without retraining.

        Loads a new dataset, removes unnamed index columns, scales the features using
        the same scaler from training, makes predictions, and displays performance
        statistics and confusion matrix.

        Args:
            test_dataset (str): Path to the CSV file containing the new test data.
        """
        df = pd.read_csv(test_dataset)

        # Remove index column if it exists (Unnamed: 0)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        y_test = df['label'].values
        X_test = df.drop(['label'], axis=1).values

        # Scale using the same scaler that was used during training
        X_test_scaled = self.scaler.transform(X_test)

        y_pred = self.model.predict(X_test_scaled)
        self.display_stats(y_test, y_pred, test_dataset, self.model_choice)
