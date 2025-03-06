import pandas
import numpy
import matplotlib.pyplot

def load_data(csv_path):
    df = pandas.read_csv(csv_path)
    print("Data shape:", df.shape)
    return df

def display_sample_images(df, num_samples=5):
    for i in range(num_samples):
        label = df.iloc[i, 0]
        img = df.iloc[i, 1:].to_numpy().reshape(28, 28)
        matplotlib.pyplot.imshow(img, cmap="gray")
        matplotlib.pyplot.title("Label: " + str(label))
        matplotlib.pyplot.axis("off")
        matplotlib.pyplot.show()

if __name__ == "__main__":
    csv_path = "data.csv"
    df = load_data(csv_path)
    display_sample_images(df, num_samples=5)