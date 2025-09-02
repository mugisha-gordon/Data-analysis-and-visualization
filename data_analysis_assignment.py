# ===============================================================
# Assignment: Data Loading, Analysis, and Visualization with Pandas & Matplotlib
# ===============================================================

# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ---------------------------------------------------------------
# Task 1: Load and Explore the Dataset
# ---------------------------------------------------------------

try:
    # Load the Iris dataset
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame
    print("✅ Dataset loaded successfully!\n")
except FileNotFoundError:
    print("❌ File not found. Please check the file path.")
except Exception as e:
    print("❌ Error loading dataset:", e)

# Display first 5 rows
print("First 5 rows of dataset:")
print(df.head(), "\n")

# Check structure, datatypes, and missing values
print("Dataset Info:")
print(df.info())

print("\nMissing Values per Column:")
print(df.isnull().sum(), "\n")

# Clean dataset (Iris has no missing values, but code included for practice)
df = df.dropna()

# ---------------------------------------------------------------
# Task 2: Basic Data Analysis
# ---------------------------------------------------------------

# Basic statistics
print("\nSummary Statistics:")
print(df.describe(), "\n")

# Grouping: mean petal length per species
grouped = df.groupby("target")["petal length (cm)"].mean()
print("Mean Petal Length by Species (target index):")
print(grouped, "\n")

# Add species names for readability
df["species"] = df["target"].map(dict(enumerate(iris_data.target_names)))

# Observations
print("🔍 Observations:")
print("1. Iris-virginica tends to have the longest petals and sepals.")
print("2. Iris-setosa has the shortest petal length and width, making it easily separable.")
print("3. There is noticeable variation in petal sizes across species, while sepal length overlaps more.\n")

# ---------------------------------------------------------------
# Task 3: Data Visualization
# ---------------------------------------------------------------

sns.set(style="whitegrid")  # nicer plot style

# 1. Line chart (trends over index for sepal length & width)
plt.figure(figsize=(8, 5))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length")
plt.plot(df.index, df["sepal width (cm)"], label="Sepal Width")
plt.title("Line Chart: Sepal Measurements over Samples")
plt.xlabel("Sample Index")
plt.ylabel("Measurement (cm)")
plt.legend()
plt.show()

# 2. Bar chart (average petal length per species)
plt.figure(figsize=(7, 5))
sns.barplot(x="species", y="petal length (cm)", data=df, ci=None, palette="Set2")
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3. Histogram (distribution of sepal length)
plt.figure(figsize=(7, 5))
plt.hist(df["sepal length (cm)"], bins=15, edgecolor="black")
plt.title("Histogram: Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot (sepal length vs petal length, color by species)
plt.figure(figsize=(7, 5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)",
                hue="species", data=df, palette="Set1")
plt.title("Scatter Plot: Sepal Length vs Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
