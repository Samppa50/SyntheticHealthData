import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess the data
data = pd.read_csv('../exampleData/diabetes.csv')
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Define the generator
def build_generator(latent_dim, output_dim):
    model = Sequential([
        Dense(128, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(256),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(output_dim, activation='sigmoid')  # Changed activation to 'sigmoid'
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

# Hyperparameters
latent_dim = 10
data_dim = data_scaled.shape[1]
epochs = 100
batch_size = 64

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
    idx = np.random.randint(0, data_scaled.shape[0], batch_size)
    real_data = data_scaled[idx]
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generated_data = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_data, real)
    d_loss_fake = discriminator.train_on_batch(generated_data, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, real)

    # Print progress
    if epoch % 1000 == 0:
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]")

# Generate synthetic data
noise = np.random.normal(0, 1, (1000, latent_dim))
synthetic_data = generator.predict(noise)
synthetic_data = scaler.inverse_transform(synthetic_data)

# Clip negative values for specific columns
columns_to_clip = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'Age', 'Pregnancies']
for col in columns_to_clip:
    col_index = data.columns.get_loc(col)
    synthetic_data[:, col_index] = np.clip(synthetic_data[:, col_index], 0, None)

# Apply additional constraints
age_index = data.columns.get_loc('Age')
synthetic_data[:, age_index] = np.clip(synthetic_data[:, age_index], 0, 70)  # Limit age to 70 years

pregnancies_index = data.columns.get_loc('Pregnancies')
synthetic_data[:, pregnancies_index] = np.clip(synthetic_data[:, pregnancies_index], 0, 10)  # Limit pregnancies to 10

# Save synthetic data
synthetic_df = pd.DataFrame(synthetic_data, columns=data.columns)
synthetic_df.to_csv('synthetic_diabetes_data.csv', index=False)
noise = np.random.normal(0, 1, (768, latent_dim))
synthetic_data = generator.predict(noise)
synthetic_data = scaler.inverse_transform(synthetic_data)

# Clip negative values for specific columns
columns_to_clip = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin']
for col in columns_to_clip:
    col_index = data.columns.get_loc(col)
    synthetic_data[:, col_index] = np.clip(synthetic_data[:, col_index], 0, None)



# Save synthetic data
synthetic_df = pd.DataFrame(synthetic_data, columns=data.columns)

# 'Pregnancies' and 'Outcome' as int
synthetic_df['Pregnancies'] = synthetic_df['Pregnancies'].round().astype(int)
synthetic_df['Outcome'] = synthetic_df['Outcome'].round().astype(int)
synthetic_df.to_csv('synthetic_diabetes_data.csv', index=False)

#testi2