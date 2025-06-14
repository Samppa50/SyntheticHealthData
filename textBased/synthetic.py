import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import os
import openpyxl
from scipy.stats import ttest_ind
import pickle
from flask import session

progress_data = {'value': 0}
stop_flag = {}

def stop_progress(session_id):
    print("Checking stop flag for session_id:", session_id)
    return stop_flag.get(session_id, False)

def request_stop(session_id):
    stop_flag[session_id] = True
    print(f"Stopping progress for session_id: {session_id}")

# Updates the progress bar
def update_progress(session_id, value):
    progress_data[session_id] = value

def get_progress(session_id):
    return progress_data.get(session_id, 0)

def set_df1(dataframe, session_id):
    #Serialize and store df1 in the session
    print(f"Setting df1 for session_id: {session_id}")
    session[f'df1_{session_id}'] = pickle.dumps(dataframe)

def get_df1(session_id):
    #Retrieve and deserialize df1 from the session.
    pickled_df1 = session.get(f'df1_{session_id}')
    if pickled_df1 is None:
        raise ValueError("df1 has not been set for this session.")
    return pickle.loads(pickled_df1)

def set_df2(dataframe, session_id):
    #Serialize and store df2 in the session.
    print(f"Setting df2 for session_id: {session_id}")
    session[f'df2_{session_id}'] = pickle.dumps(dataframe)

def get_df2(session_id):
    #Retrieve and deserialize df2 from the session.
    pickled_df2 = session.get(f'df2_{session_id}')
    if pickled_df2 is None:
        raise ValueError("df2 has not been set for this session.")
    return pickle.loads(pickled_df2)


def main(session_id ,name):
    # Load and preprocess the data
    if name.endswith('.xlsx'):
        df = pd.read_excel('Files/uploads/'+ session_id+ '/' + name, engine=None)
        filename = os.path.splitext('Files/uploads/' + session_id + '/' + name)[0]
        csv_name = filename + ".csv"
        df.to_csv(csv_name, index=False)
        name = os.path.basename(csv_name)
        print(name)

        data = pd.read_csv(csv_name)
    else:
        data = pd.read_csv('Files/uploads/'+ session_id+ '/' + name)


    #Converting non numeric values into numbers
    col_names = list(data.columns)
    for col in data.columns:
        non_numeric_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in non_numeric_cols:
        data[col] = data[col].astype('category').cat.codes
    return col_names

    #Generating the synthetic file
def generate_file(col_values, line_amount, epoch_amount, name, session_id):


    data = pd.read_csv('Files/uploads/'+ session_id+ '/' + name, encoding="ISO-8859-1", on_bad_lines='skip')

    df_decimal_source = pd.read_csv('Files/uploads/'+ session_id+ '/' + name, dtype=str, encoding="ISO-8859-1", on_bad_lines='skip')

    #counts the decimals that are used in the synthetic data

    def count_decimals(value):
        if pd.isna(value) or '.' not in value:
            return 0
        return len(value.split('.')[-1])

    decimal_places = df_decimal_source.applymap(count_decimals).max()

    # Store mappings for categorical columns
    category_mappings = {}

    non_numeric_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in non_numeric_cols:
        data[col] = data[col].astype('category')
        category_mappings[col] = dict(enumerate(data[col].cat.categories))
        data[col] = data[col].cat.codes


    non_numeric_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in non_numeric_cols:
        data[col] = data[col].astype('category').cat.codes

    non_numeric_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in non_numeric_cols:
        data[col] = data[col].astype('category').cat.codes

    exclude_columns = []
    col_ignore_zero = []
    col_bool = []

    for i in range(len(col_values)//3):
        col_bool.append(col_values[i])

    for i in range(len(col_values)//3, len(col_values)//3*2):
        col_ignore_zero.append(col_values[i])

    for i in range(len(col_values)//3*2, len(col_values)):
        exclude_columns.append(col_values[i])

    print(f"col_bool: {col_bool}")
    print(f"col_ignore_zero: {col_ignore_zero}")
    print(f"exclude_columns: {exclude_columns}")

    exclude_columns = np.array(list(map(int, exclude_columns)))
    col_ignore_zero = np.array(list(map(int, col_ignore_zero)))
    col_bool = np.array(list(map(int, col_bool)))

    col_ignore_zero = col_ignore_zero[exclude_columns == 0]
    col_bool = col_bool[exclude_columns == 0]

    print(f"col_bool: {col_bool}")
    print(f"col_ignore_zero: {col_ignore_zero}")
    print(f"exclude_columns: {exclude_columns}")

    mask = np.array(list(map(int, exclude_columns)))
    invert_mask =  1 -mask
    keep_mask = invert_mask.astype(bool)
    data = data.iloc[:, keep_mask]

    ignore_zero = [col for col, flag in zip(data.columns, list(map(int, col_ignore_zero))) if flag == 1]
    data[ignore_zero] = data[ignore_zero].replace(0, np.nan)
    data = data.dropna(subset=ignore_zero)

    # Replaces any blank values with nan values and then turns the nan values into 0
    data.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    data.fillna(0, inplace=True)

    # Load the datasets
    real_df = data

    set_df1(data, session_id)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Hyperparameters
    latent_dim = 10
    data_dim = data_scaled.shape[1]
    epochs = epoch_amount
    batch_size = 64

    # Define the generator
    def build_generator(latent_dim, output_dim):
        model = Sequential([
            Dense(128, input_dim=latent_dim),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dense(256),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dense(output_dim, activation='sigmoid')
        ])
        return model

    # Define the discriminator
    def build_discriminator(input_dim):
        model = Sequential([
            Dense(256, input_dim=input_dim),
            LeakyReLU(alpha=0.2),
            Dense(128),
            LeakyReLU(alpha=0.2),
            Dense(1, activation='sigmoid')
        ])
        return model

    # GAN model
    def build_gan(generator, discriminator):
        discriminator.trainable = False
        gan_input = Input(shape=(latent_dim,))
        generated_data = generator(gan_input)
        gan_output = discriminator(generated_data)
        gan = Model(gan_input, gan_output)
        return gan


    # Build and compile models
    generator = build_generator(latent_dim, data_dim)
    discriminator = build_discriminator(data_dim)
    discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

    gan = build_gan(generator, discriminator)
    gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

    # Training the GAN
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        if stop_progress(session_id):
            print("Generation stopped by user request.")
            return None
        # Train discriminator
        discriminator.trainable = True
        idx = np.random.randint(0, data_scaled.shape[0], batch_size)
        real_data = data_scaled[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_data = generator.predict(noise)


        d_loss_real = discriminator.train_on_batch(real_data, real)
        d_loss_fake = discriminator.train_on_batch(generated_data, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        discriminator.trainable = False
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real)

        # Update progress
        progress = (epoch + 1) / epochs * 100
        update_progress(session_id, progress)

        # Print progress
        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]")

    # Generate synthetic data
    noise = np.random.normal(0, 1, (line_amount, latent_dim))
    synthetic_data = generator.predict(noise)
    synthetic_data = scaler.inverse_transform(synthetic_data)

    # Save synthetic data
    synthetic_df = pd.DataFrame(synthetic_data, columns=data.columns)

    # Modifing synthetic data into the desired form
    convert_bool = [col for col, flag in zip(data.columns, list(map(int,col_bool))) if flag == 1]

    synthetic_df[convert_bool] = synthetic_df[convert_bool].round().astype(int)

     #rounding the synthetic values with the right decimal amounts
    for col in synthetic_df.columns:
        if col in decimal_places:
            try:
                decimals = int(decimal_places[col])
                synthetic_df[col] = pd.to_numeric(synthetic_df[col], errors='coerce')
                synthetic_df[col] = (
                    synthetic_df[col].round(decimals).astype(int)
                    if decimals == 0 else
                    synthetic_df[col].round(decimals)
                )
            except ValueError:
                pass

    # Save the synthetic DataFrame to df2
    set_df2(synthetic_df, session_id)

    # Convert numeric values back to original categorical values
    for col, mapping in category_mappings.items():
        print(f"{col}: {mapping}")
        print(synthetic_df.head())
        if col in synthetic_df.columns:
            non_reversed_mapping = {k: v for k, v in mapping.items()}
            synthetic_df[col] = synthetic_df[col].map(non_reversed_mapping)



    # Perform t-tests for each numeric column
    synthetic_df_ttest = synthetic_df


    t_test_results = {}
    for column in synthetic_df_ttest.columns:
        if pd.api.types.is_numeric_dtype(synthetic_df_ttest[column]):
            t_stat, p_value = ttest_ind(synthetic_df_ttest[column], real_df[column])
            t_test_results[column] = {'t_statistic': t_stat, 'p_value': p_value}

    # Display the results
    for column, result in t_test_results.items():
        print(f"{column}: t_statistic = {result['t_statistic']:.4f}, p_value = {result['p_value']:.4e}")



    # saving the synthetic file
    synthetic_name = "synthetic_" + os.path.splitext(name)[0] + ".csv"
    path = 'Files/downloads/'+ session_id + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    synthetic_df.to_csv(os.path.join(path, synthetic_name), index=False)

    return synthetic_name