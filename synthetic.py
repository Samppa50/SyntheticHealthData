import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import os
import openpyxl



def main(name):
    # Load and preprocess the data
    if name.endswith('.xlsx'):
        df = pd.read_excel('Files/uploads/' + name, engine=None)
        filename = os.path.splitext('Files/uploads/' + name)[0]
        csv_name = filename + ".csv"
        df.to_csv(csv_name, index=False)
        name = os.path.basename(csv_name)

        data = pd.read_csv(csv_name)
    else:
        data = pd.read_csv('Files/uploads/' + name)

    #Converting non numeric values into numbers
    col_names = list(data.columns)
    for col in data.columns:
        non_numeric_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in non_numeric_cols:
        data[col] = data[col].astype('category').cat.codes
    return col_names

    #Generating the synthetic file
def generate_file(col_values, line_amount, epoch_amount, name):
    data = pd.read_csv('Files/uploads/' + name, encoding="ISO-8859-1", on_bad_lines='skip')

    df_decimal_source = pd.read_csv('Files/uploads/' + name, dtype=str)
    
    def count_decimals(value):
        if pd.isna(value) or '.' not in value:
            return 0
        return len(value.split('.')[-1])

    decimal_places = df_decimal_source.applymap(count_decimals).max()


    non_numeric_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in non_numeric_cols:
        data[col] = data[col].astype('category').cat.codes

    non_numeric_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in non_numeric_cols:
        data[col] = data[col].astype('category').cat.codes

    col_ignore_zero = []
    col_bool = []

    for i in range(len(col_values)//2):
        col_bool.append(col_values[i])

    for i in range(len(col_values)//2, len(col_values)):
        col_ignore_zero.append(col_values[i])

    ignore_zero = [col for col, flag in zip(data.columns, list(map(int, col_ignore_zero))) if flag == 1]
    data[ignore_zero] = data[ignore_zero].replace(0, np.nan)
    data = data.dropna(subset=ignore_zero)


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
    
    #Rounding the synthetic values with the right decimal amounts
    for col in synthetic_df.columns:
        if col in decimal_places:
            try:
                decimals = int(decimal_places[col])
                synthetic_df[col] = pd.to_numeric(synthetic_df[col], errors='coerce').round(decimals)
            except ValueError:
                pass 


    synthetic_name = "synthetic_"+name
    path = 'Files/downloads/'
    if not os.path.exists(path):
        os.makedirs(path)
    synthetic_df.to_csv(os.path.join(path, synthetic_name), index=False)

    return synthetic_name