import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Ensure Date column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Basic statistics
    print(df.info())
    print(df.describe())

    # Dual Y-Axis Plot for Temperature and Growth Rate
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Temperature on the primary y-axis
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature', color='blue')
    ax1.plot(df['Date'], df['Temperature'], color='blue', label='Temperature')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Growth Rate on the secondary y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Growth Rate', color='orange')
    ax2.plot(df['Date'], df['GrowthRate'], color='orange', label='Growth Rate')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Improve x-axis readability
    ax1.xaxis.set_major_locator(plt.MaxNLocator(10))  # Show fewer ticks on the x-axis
    plt.xticks(rotation=45, fontsize=10)             # Rotate labels for better readability

    # Add title
    fig.suptitle("Temperature vs Growth Rate Trends")

    # Adjust layout to prevent label overlap
    fig.tight_layout()

    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

if __name__ == "__main__":
    analyze_data("./data/mock_poultry_data.csv")
