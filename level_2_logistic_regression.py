import pandas
import matplotlib.pyplot
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def load_data(csv_path):
    df = pandas.read_csv(csv_path)
    return df

def prepare_data(df):
    y = df["label"].to_numpy()
    X = df.drop("label", axis=1).to_numpy().astype("float32")
    X = X / 255.0
    return X, y

def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Logistic Regression Accuracy:", acc)
    cm = confusion_matrix(y_test, y_pred)
    matplotlib.pyplot.figure(figsize=(8,6))
    seaborn.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    matplotlib.pyplot.title("Confusion Matrix - Logistic Regression")
    matplotlib.pyplot.xlabel("Predicted")
    matplotlib.pyplot.ylabel("Actual")
    matplotlib.pyplot.show()
    num_classes = model.coef_.shape[0]
    matplotlib.pyplot.figure(figsize=(15,3))
    for i in range(num_classes):
        coef = model.coef_[i].reshape(28,28)
        matplotlib.pyplot.subplot(2,5, i+1)
        matplotlib.pyplot.imshow(coef, cmap="viridis")
        matplotlib.pyplot.title("Coef for class " + str(i))
        matplotlib.pyplot.axis("off")
    matplotlib.pyplot.suptitle("Logistic Regression Coefficients")
    matplotlib.pyplot.show()
    return model

if __name__ == "__main__":
    csv_path = "data.csv"
    df = load_data(csv_path)
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_logistic_regression(X_train, y_train, X_test, y_test)
