import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from synthetic import real_df, synthetic_df

def correlation():
    
    df1 = pd.read_csv(real_df)
    df2 = pd.read_csv(synthetic_df)

    # Drop the first column (index column)
    df1 = df1.drop(df1.columns[0], axis=1)
    df2 = df2.drop(df2.columns[0], axis=1)

    # Fill missing values with the mean of each column
    df1 = df1.fillna(df1.mean())
    df2 = df2.fillna(df2.mean())

    # Compute the correlation matrices
    corr_matrix1 = df1.corr()
    corr_matrix2 = df2.corr()

    # Plot the heatmaps
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.heatmap(corr_matrix1, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix: orginal.csv')

    plt.subplot(1, 2, 2)
    sns.heatmap(corr_matrix2, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix: synthetic.csv')

    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300) #dpi will adjust image resolution
    plt.show()
