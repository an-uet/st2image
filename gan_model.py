# %tensorflow_version 1.x
import ast
import os

from tqdm import tqdm_notebook, tqdm
import matplotlib
import matplotlib.pylab as plt
from math import ceil
from PIL import Image
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, ReLU, LeakyReLU, Dense
from keras.layers.core import Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
from keras.datasets import cifar10
from keras.initializers import RandomNormal


import pandas as pd

from sklearn.model_selection import train_test_split


def process_sub_image(image_path):
    # Load the image
    image = Image.open(image_path)

    image = image.resize((256, 256))

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Normalize the pixel values to the range [-1, 1]
    image_array = 2 * (image_array / 255) - 1

    image_array = np.expand_dims(image_array, axis=0)

    return image_array[0]

def scale_matrix(matrix, max):
    scaled_matrix = 2 * matrix/max - 1
    return scaled_matrix


import re
def process_gene_expression(gene_expession):
    gene_expression = re.sub('\[|\]', '', gene_expession).strip()
    gene_expession = gene_expession.split(', ')
    gene_expession = [float(re.sub('\[|\]', '', i)) for i in gene_expession]
    return gene_expession

def load_data(data_path='data.csv'):
    image_shape=(256, 256, 3)
    df = pd.read_csv(data_path)
    # df = df[df['patient'] == 'A']
    df['image'] = [process_sub_image(row['image_path']) for i, row in df.iterrows()]
    
    X = df[['Unnamed: 0', 'image']].values
    y = df['patient'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
    X_train = X

    # Accessing the image column and reshaping it to the desired shape
    X_train_images = np.array([img.reshape(image_shape) for img in X_train[:, 1]])
    X_test_images = np.array([img.reshape(image_shape) for img in X_test[:, 1]])

    # Accessing the index column
    index_train = X_train[:, 0]  
    index_test = X_test[:, 0]    

    gene_expression_train = df.iloc[index_train]['gene_expression']
    gene_expression_train = [ast.literal_eval(row) for row in gene_expression_train]
    # gene_expression_train = [process_gene_expression(row) for row in gene_expression_train]

    gene_expression_test = df.iloc[index_test]['gene_expression']
    # gene_expression_test = [ast.literal_eval(row) for row in gene_expression_test]
    # gene_expression_test = [process_gene_expression(row) for row in gene_expression_test]

    print(X_train_images.shape)
    return X_train_images, np.array(gene_expression_train), X_test_images, gene_expression_test




def plot_images(images, filename):
    h, w, c = images.shape[1:]
    grid_size = ceil(np.sqrt(images.shape[0]))
    images = (images + 1) / 2. * 255.
    images = images.astype(np.uint8)
    images = (images.reshape(grid_size, grid_size, h, w, c)
              .transpose(0, 2, 1, 3, 4)
              .reshape(grid_size*h, grid_size*w, c))
    #plt.figure(figsize=(16, 16))
    plt.imsave(filename, images)
    plt.imshow(images)
    plt.show()

def plot_losses(losses_d, losses_g, filename):
    fig, axes = plt.subplots(1, 2, figsize=(8, 2))
    axes[0].plot(losses_d)
    axes[1].plot(losses_g)
    axes[0].set_title("losses_d")
    axes[1].set_title("losses_g")
    plt.tight_layout()
    plt.savefig(filename)
    #plt.close()



def build_generator(z_dim=128, n_filter=128):
    init = RandomNormal(stddev=0.02)

    G = Sequential()
    G.add(Dense(2 * 2 * n_filter * 4, input_shape=(z_dim,), use_bias=True, kernel_initializer=init))

    # 2*2*512
    G.add(Reshape((2, 2, n_filter * 4)))
    G.add(BatchNormalization())
    G.add(LeakyReLU(0.2))

    # 4*4*256
    G.add(Conv2DTranspose(n_filter * 2, kernel_size=(5, 5), strides=2, padding='same', use_bias=True,
                          kernel_initializer=init))
    G.add(BatchNormalization())
    G.add(LeakyReLU(0.2))

    # 8*8*128
    G.add(Conv2DTranspose(n_filter * 1, kernel_size=(5, 5), strides=2, padding='same', use_bias=True,
                          kernel_initializer=init))
    G.add(BatchNormalization())
    G.add(LeakyReLU(0.2))

    # 16*16*64
    G.add(Conv2DTranspose(n_filter // 2, kernel_size=(5, 5), strides=2, padding='same', use_bias=True,
                          kernel_initializer=init))
    G.add(BatchNormalization())
    G.add(LeakyReLU(0.2))

    # 32*32*32
    G.add(Conv2DTranspose(n_filter // 4, kernel_size=(5, 5), strides=2, padding='same', use_bias=True,
                          kernel_initializer=init))
    G.add(BatchNormalization())
    G.add(LeakyReLU(0.2))

    # 64*64*16
    G.add(Conv2DTranspose(n_filter // 8, kernel_size=(5, 5), strides=2, padding='same', use_bias=True,
                          kernel_initializer=init))
    G.add(BatchNormalization())
    G.add(LeakyReLU(0.2))

    # 128*128*8
    G.add(Conv2DTranspose(n_filter // 16, kernel_size=(5, 5), strides=2, padding='same', use_bias=True,
                          kernel_initializer=init))
    G.add(BatchNormalization())
    G.add(LeakyReLU(0.2))

    # 256*256*3
    G.add(Conv2DTranspose(3, kernel_size=(5, 5), strides=2, padding='same', use_bias=True, kernel_initializer=init))
    G.add(BatchNormalization())
    G.add(Activation('tanh'))

    print('Build Generator')
    print(G.summary())

    return G


def build_discriminator(input_shape=(256, 256, 3), n_filter=128):
    init = RandomNormal(stddev=0.02)

    D = Sequential()

    # 128*128*8
    D.add(Conv2D(n_filter // 16, input_shape=input_shape, kernel_size=(5, 5), strides=2, padding='same', use_bias=True,
                 kernel_initializer=init))
    D.add(LeakyReLU(0.2))

    # 64*64*16
    D.add(Conv2D(n_filter // 8, input_shape=input_shape, kernel_size=(5, 5), strides=2, padding='same', use_bias=True,
                 kernel_initializer=init))
    D.add(LeakyReLU(0.2))

    # 32*32*32
    D.add(Conv2D(n_filter // 4, input_shape=input_shape, kernel_size=(5, 5), strides=2, padding='same', use_bias=True,
                 kernel_initializer=init))
    D.add(LeakyReLU(0.2))

    # 16*16*64
    D.add(Conv2D(n_filter // 2, input_shape=input_shape, kernel_size=(5, 5), strides=2, padding='same', use_bias=True,
                 kernel_initializer=init))
    D.add(LeakyReLU(0.2))

    # 8*8*128
    D.add(Conv2D(n_filter, kernel_size=(5, 5), strides=2, padding='same', use_bias=True, kernel_initializer=init))
    D.add(BatchNormalization())
    D.add(LeakyReLU(0.2))

    # 4*4*256
    D.add(Conv2D(n_filter * 2, kernel_size=(5, 5), strides=2, padding='same', use_bias=True, kernel_initializer=init))
    D.add(BatchNormalization())
    D.add(LeakyReLU(0.2))

    # 2*2*512
    D.add(Conv2D(n_filter * 4, kernel_size=(5, 5), strides=2, padding='same', use_bias=True, kernel_initializer=init))
    D.add(BatchNormalization())
    D.add(LeakyReLU(0.2))

    D.add(Flatten())
    D.add(Dense(1, kernel_initializer=init))
    D.add(Activation('sigmoid'))

    print('Build discriminator')
    print(D.summary())

    return D



def train(n_filter=128, z_dim=100, lr_d=2e-4, lr_g=2e-4, epochs=500, batch_size=128,
          epoch_per_checkpoint=1, n_checkpoint_images=36, verbose=10):
    X_train, gene_expression,  X_test, gene_expression_test = load_data('data.csv')
    image_shape = X_train[0].shape
    print('Image shape {}, min val {}, max val {}'.format(image_shape, np.min(X_train[0]), np.max(X_train[0])))
    print('gene expression shape {}, min val {}, max val {}'.format(gene_expression[0].shape, np.min(gene_expression[0]), np.max(gene_expression[0])))
    plot_images(X_train[:n_checkpoint_images], 'fake_image_100/real_image.png')

    # Build model
    G = build_generator(z_dim, n_filter)
    D = build_discriminator(image_shape, n_filter)

    # Loss for discriminator
    D.compile(Adam(learning_rate=lr_d, beta_1=0.5), loss='binary_crossentropy', metrics=['binary_accuracy'])

    # D(G(X))
    D.trainable = False
    z = Input(shape=(z_dim,))
    D_of_G = Model(inputs=z, outputs=D(G(z)))

    # Loss for generator
    D_of_G.compile(Adam(learning_rate=lr_d, beta_1=0.5), loss='binary_crossentropy', metrics=['binary_accuracy'])

    # Labels for computing the losses
    real_labels = np.ones(shape=(batch_size, 1))
    fake_labels = np.zeros(shape=(batch_size, 1))
    losses_d, losses_g = [], []

    # fix a z vector for training evaluation
    # z_fixed = np.random.uniform(-1, 1, size=(n_checkpoint_images, z_dim))

    gene_expression_fixed = gene_expression[:n_checkpoint_images]
    max_val = np.max(gene_expression)
    min_val = np.min(gene_expression)

    # # Step 2: Scale array to range [0, 1]
    # gene_expression_scaled = (gene_expression_fixed  - min_val) / (max_val - min_val)

    # # Step 3: Center values around 0
    # mean_val = np.mean(gene_expression_scaled)
    # gene_expression_centered = gene_expression_scaled - mean_val

    # # Step 4: Scale values to range [-1, 1]
    # gene_expression_normalized = gene_expression_centered * 2

    # # z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
    # # print('z_shape ', z.shape)
    # z_fixed = gene_expression_normalized
    z_fixed = scale_matrix(gene_expression_fixed, max_val)

    X_train  = X_train[n_checkpoint_images:]
    
    losses = []
    print('X_train.shape[0] : ',  X_train.shape[0])
    for e in tqdm(range(1, epochs + 1)):
        n_steps = X_train.shape[0] // batch_size
        for i in range(n_steps):
            # Train discriminator
            D.trainable = True
            real_images = X_train[i * batch_size:(i + 1) * batch_size]

            loss_d_real = D.train_on_batch(x=real_images, y=real_labels)[0]

            gene_expression_batch = gene_expression[i * batch_size:(i + 1) * batch_size]
            # max_val = np.max(gene_expression)
            # min_val = np.min(gene_expression)

            # # Step 2: Scale array to range [0, 1]
            # gene_expression_batch_scaled = (gene_expression_batch - min_val) / (max_val - min_val)

            # # Step 3: Center values around 0
            # gene_expression_batch_centered = gene_expression_batch_scaled - np.mean(gene_expression_batch_scaled)

            # # Step 4: Scale values to range [-1, 1]
            # gene_expression_batch_normalized = gene_expression_batch_centered * 2

            # # z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
            # # print('z_shape ', z.shape)
            z = scale_matrix(gene_expression_batch, max_val)
            # print('z_shape ', z.shape)
            fake_images = G.predict_on_batch(z)
            loss_d_fake = D.train_on_batch(x=fake_images, y=fake_labels)[0]

            loss_d = loss_d_real + loss_d_fake

            # Train generator

            D.trainable = False
            loss_g = D_of_G.train_on_batch(x=z, y=real_labels)[0]

            losses_d.append(loss_d)
            losses_g.append(loss_g)
            # losses.append((loss_d, loss_g))
            print(loss_d, loss_g)
            if i == 0 and e % verbose == 0:
                print('Epoch {}'.format(e))
                fake_images = G.predict(z_fixed)
                # print("\tPlotting images and losses")
                plot_images(fake_images, "fake_image_100/fake_images_e_{}.png".format(e))
                plot_losses(losses_d, losses_g, "fake_image_100/losses.png")
                # losses = pd.DataFrame(losses, columns=['loss_d', 'loss_g'])
                # losses.to_csv('fake_image_100/train_loss.csv')
        if e % 50 == 0:
            # Save the model checkpoint
            G.save("fake_image_100/G_model_epoch_{}.h5".format(e))
            D.save("fake_image_100/D_model_epoch_{}.h5".format(e))
    
    

if __name__ == '__main__':
    train()
