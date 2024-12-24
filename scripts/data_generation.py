import pandas as pd
import numpy as np

def generate_data(file_path):
    np.random.seed(42)

    data = {
        'Date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
        'FarmID': np.random.randint(1, 5, size=100),
        'Temperature': np.sin(np.linspace(0, 3 * np.pi, 100)) * 5 + 25 + np.random.uniform(-2, 2, 100),
        'Humidity': np.random.uniform(50, 80, size=100),
        'GrowthRate': np.random.uniform(1.0, 2.0, size=100),
        'BehaviorScore': np.random.randint(5, 10, size=100),
        'MortalityRate': np.random.uniform(0.01, 0.1, size=100)
    }

    df = pd.DataFrame(data)
    df['WelfareScore'] = df['BehaviorScore'] - (df['MortalityRate'] * 10)
    df.to_csv(file_path, index=False)
    print(f"Mock data saved to {file_path}")

if __name__ == "__main__":
    generate_data("./data/mock_poultry_data.csv")
