import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Load and preprocess images
def load_and_preprocess_images(dataset_path, image_size=(64, 64)):
    X, y = [], []
    class_labels = sorted(os.listdir(dataset_path))
    for label in class_labels:
        class_folder = os.path.join(dataset_path, label)
        if not os.path.isdir(class_folder):
            continue
        for fn in os.listdir(class_folder):
            if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                img = cv2.imread(os.path.join(class_folder, fn))
                img = cv2.resize(img, image_size)
                X.append(img.flatten())
                y.append(label)
    return np.array(X), np.array(y)

# Plot Confusion Matrix
def plot_conf_matrix(y_true, y_pred, class_names, title,color):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=color,
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f"{title} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# Naive Bayes
def run_naive_bayes(X_train, X_test, y_train, y_test, class_names):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\n=== Naive Bayes ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Report:\n", classification_report(y_test, y_pred))
    plot_conf_matrix(y_test, y_pred, class_names, "Naive Bayes","Blues")

# Decision Tree 
def run_decision_tree(X_train, X_test, y_train, y_test, class_names):
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\n=== Decision Tree ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Report:\n", classification_report(y_test, y_pred))
    plot_conf_matrix(y_test, y_pred, class_names, "Decision Tree","Greens")
    return clf

# MLP 
def run_mlp(X_train, X_test, y_train, y_test, class_names):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    print("\n=== MLP Classifier ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Report:\n", classification_report(y_test, y_pred))
    plot_conf_matrix(y_test, y_pred, class_names, "MLP Classifier","Reds")

if __name__ == "__main__":
    dataset_path = "Vehicles"
    X, y_str = load_and_preprocess_images(dataset_path)
    print("Dataset shape:", X.shape)
    print("Raw classes:", sorted(set(y_str)))

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    print("Encoded classes:", list(le.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    class_names = le.classes_.tolist()

    run_naive_bayes(X_train, X_test, y_train, y_test, class_names)
    dt_model = run_decision_tree(X_train, X_test, y_train, y_test, class_names)
    run_mlp(X_train, X_test, y_train, y_test, class_names)

    # Plot decision tree
    plt.figure(figsize=(12, 6))
    plot_tree(
        dt_model,
        max_depth=2,
        filled=True,
        feature_names=None,
        class_names=class_names
    )
    plt.title("Decision Tree (Top 2 Levels)")
    plt.show()
