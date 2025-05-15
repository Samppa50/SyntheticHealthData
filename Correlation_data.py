import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from synthetic import get_df1, get_df2


def correlation(new_file, original_file, session_id):

    df1 = get_df1(session_id, original_file)
    df2 = get_df2(session_id)

    #df1 = pd.read_csv('Files/uploads/' + session_id + '/' + original_file)
    #df2 = pd.read_csv('Files/downloads/' + session_id + '/' + new_file)

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
    location = 'static/images/correlation_matrix.png'
    plt.savefig(location, dpi=80)
    plt.close()

    return location