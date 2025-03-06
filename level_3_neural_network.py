import pandas
import matplotlib.pyplot
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def load_data(csv_path):
    df = pandas.read_csv(csv_path)
    return df

def prepare_data(df):
    y = df["label"].to_numpy()
    X = df.drop("label", axis=1).to_numpy().astype("float32")
    X = X / 255.0
    return X, y

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_cnn(X_train, y_train, X_test, y_test, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    X_train_cnn = X_train.reshape(-1, 1, 28, 28)
    X_test_cnn = X_test.reshape(-1, 1, 28, 28)
    X_train_tensor = torch.tensor(X_train_cnn, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_cnn, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    num_classes = len(torch.unique(y_train_tensor))
    model = CNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=(device.type=="cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        train_loss = running_loss / total
        train_acc = correct / total
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast(device_type="cuda", enabled=(device.type=="cuda")):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()
        val_loss /= total_val
        val_acc = correct_val / total_val
        print("Epoch " + str(epoch+1) + "/" + str(num_epochs) + ": Train Loss " + str(round(train_loss, 4)) + ", Train Acc " + str(round(train_acc, 4)) + ", Val Loss " + str(round(val_loss, 4)) + ", Val Acc " + str(round(val_acc, 4)))
    print("Final Test Accuracy (Validation):", val_acc)
    return model

def display_predictions(model, X_test, y_test):
    device = next(model.parameters()).device
    model.eval()
    X_test_cnn = X_test.reshape(-1, 1, 28, 28)
    X_test_tensor = torch.tensor(X_test_cnn, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.cpu().numpy()
    num_samples = 10
    matplotlib.pyplot.figure(figsize=(15,4))
    for i in range(num_samples):
        img = X_test[i].reshape(28,28)
        matplotlib.pyplot.subplot(1, num_samples, i+1)
        matplotlib.pyplot.imshow(img, cmap="gray")
        matplotlib.pyplot.title("Actual: " + str(y_test[i]) + "\nPred: " + str(pred_labels[i]))
        matplotlib.pyplot.axis("off")
    matplotlib.pyplot.suptitle("Sample Predictions from CNN")
    matplotlib.pyplot.show()

if __name__ == "__main__":
    csv_path = "data.csv"
    df = load_data(csv_path)
    X, y = prepare_data(df)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_cnn(X_train, y_train, X_test, y_test, num_epochs=10)
    display_predictions(model, X_test, y_test)
