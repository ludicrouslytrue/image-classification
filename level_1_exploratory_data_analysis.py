import pandas
import matplotlib.pyplot
import seaborn

def load_data(csv_path):
    df = pandas.read_csv(csv_path)
    return df

def perform_eda(df):
    print("Data shape:", df.shape)
    print("Summary statistics:")
    print(df.describe())
    matplotlib.pyplot.figure(figsize=(8,6))
    seaborn.countplot(x="label", data=df, hue="label", palette="pastel", dodge=False)
    matplotlib.pyplot.legend([], [], frameon=False)
    matplotlib.pyplot.title("Distribution of Labels")
    matplotlib.pyplot.xlabel("Label")
    matplotlib.pyplot.ylabel("Count")
    matplotlib.pyplot.show()
    unique_labels = sorted(df["label"].unique())
    matplotlib.pyplot.figure(figsize=(12,8))
    for i, label in enumerate(unique_labels):
        sample = df[df["label"] == label].iloc[0, 1:]
        img = sample.to_numpy().reshape(28,28)
        matplotlib.pyplot.subplot(2,5, i+1)
        matplotlib.pyplot.imshow(img, cmap="gray")
        matplotlib.pyplot.title("Label: " + str(label))
        matplotlib.pyplot.axis("off")
    matplotlib.pyplot.suptitle("Sample Images from Each Category")
    matplotlib.pyplot.show()
    sample_pixels = df.iloc[:100, 1:].values.flatten()
    matplotlib.pyplot.figure(figsize=(8,6))
    matplotlib.pyplot.hist(sample_pixels, bins=50, color="blue", alpha=0.7)
    matplotlib.pyplot.title("Histogram of Pixel Values (first 100 images)")
    matplotlib.pyplot.xlabel("Pixel Intensity")
    matplotlib.pyplot.ylabel("Frequency")
    matplotlib.pyplot.show()

if __name__ == "__main__":
    csv_path = "data.csv"
    df = load_data(csv_path)
    perform_eda(df)
