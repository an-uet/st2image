import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import warnings
import os
import shutil


def split_image(image, x, y, pixel_x, pixel_y, folder_path):
    tensor_image = transforms.ToTensor()(image)
    _, height, width = tensor_image.size()
    #     print(height, width)
    sub_image_size = (height / 35, width / 33)

    # Calculate the bounding box for the sub-image
    left = pixel_x - sub_image_size[0]
    top = pixel_y - sub_image_size[1]
    right = pixel_x + sub_image_size[0]
    bottom = pixel_y + sub_image_size[1]

    # Crop the image using the calculated bounding box
    sub_image = image.crop((left, top, right, bottom))

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256
        transforms.ToTensor()  # Convert to PyTorch tensor
    ])

    # Apply the transformation
    resized_image = transform(sub_image)

    # Convert tensor back to image and save
    resized_image = transforms.ToPILImage()(resized_image)

    resized_image.save(f"{folder_path}/sub_img_{int(x)}_{int(y)}.jpg")


def process_image(image_path, spotfile_path, folder_path):
    # Convert the image to a PyTorch tensor
    image = Image.open(image_path)
    df = pd.read_csv(spotfile_path, delimiter='\t')
    for i, row in tqdm(df.iterrows()):
        x = row['x']
        y = row['y']
        x_pixel = row['pixel_x']
        y_pixel = row['pixel_y']
        split_image(image, x, y, x_pixel, y_pixel, folder_path)


def main_split_image():
    # Suppress DecompressionBombWarning
    warnings.simplefilter("ignore", Image.DecompressionBombWarning)

    # List all files in the directory
    directory = 'data/ST-imgs/'
    files = os.listdir(directory)
    for patient in files:
        # path save sub-image
        folder_path = f"ST-imgs-split/{patient}"
        os.makedirs(folder_path)
        if patient in ('A', 'B', 'C', 'D'):
            for i in range(1, 7):
                sub_folder_path = f"ST-imgs-split/{patient}/{patient}{i}"
                os.makedirs(sub_folder_path)

                file_path = directory + patient + '/' + f'{patient}{i}'
                image_path = file_path + '/' + os.listdir(file_path)[0]

                spotfile_path = f'data/ST-spotfiles/{patient}{i}_selection.tsv'
                print(image_path)
                process_image(image_path, spotfile_path, sub_folder_path)

        else:
            for i in range(1, 4):
                sub_folder_path = f"ST-imgs-split/{patient}/{patient}{i}"
                os.makedirs(sub_folder_path)

                file_path = directory + patient + '/' + f'{patient}{i}'
                image_path = file_path + '/' + os.listdir(file_path)[0]
                spotfile_path = f'data/ST-spotfiles/{patient}{i}_selection.tsv'
                print(image_path)
                process_image(image_path, spotfile_path, sub_folder_path)


def find_top_gene(N=100):
    # cnts_directory = 'data/ST-cnts/'
    # files = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    # files = ['A']
    # cnts_df = []
    # for patient in files:
    #     if patient in ('A', 'B', 'C', 'D'):
    #         for i in range(1, 7):
    #             cnts_file = cnts_directory + f"{patient}{i}.tsv/" + f'ut_{patient}{i}_stdata_filtered.tsv'
    #             print(cnts_file)
    #             df = pd.read_csv(cnts_file, delimiter='\t')
    #             cnts_df.append(df)
    #     else:
    #         for i in range(1, 4):
    #             cnts_file = cnts_directory + f"{patient}{i}.tsv/" + f'ut_{patient}{i}_stdata_filtered.tsv'
    #             print(cnts_file)
    #             df = pd.read_csv(cnts_file, delimiter='\t')
    #             cnts_df.append(df)

    # result = cnts_df[0]
    # result = result.drop(columns=['Unnamed: 0'])

    # for i in range(1, len(cnts_df)):
    #     df = cnts_df[i]
    #     df = df.drop(columns=['Unnamed: 0'])
    #     result = result.add(df, fill_value=0)

    # variations = result.var()

    # # Get the top N columns with the maximum variance
    # top_max_var_gene = list(variations.nlargest(N).index)
    # top_max_var_gene.append('Unnamed: 0')
    # return top_max_var_gene
    df = pd.read_csv('full_gene_expression.csv')
    df = df.fillna(0)
    df = df.drop(['Unnamed: 0', 'patient'], axis=1)
    variations = df.var()
    top_max_var_gene = list(variations.nlargest(N).index)
    top_max_var_gene.append('Unnamed: 0')
    return top_max_var_gene

# Function to calculate the logarithm
def calculate_log(row):
    return np.log2((row * 1000) / row.sum() + 1)

def norm_gene(df):
    print('normalize gene count......')
    tmp_df = df.copy()
    df = df.drop(['Unnamed: 0'], axis=1)
    result = df.apply(lambda row: calculate_log(row), axis=1)
    result['Unnamed: 0'] = tmp_df['Unnamed: 0']
    print('normalize gene count done!')
    return result


def extract_top_gene(gene_file_path, top_max_var_gene):
    df = pd.read_csv(gene_file_path, delimiter='\t')
    df = norm_gene(df)
    # Extract column names from the DataFrame
    existing_columns = df.columns

    # Specify the desired columns
    desired_columns = top_max_var_gene

    # Find missing columns
    missing_columns = [col for col in desired_columns if col not in existing_columns]

    # Fill missing columns with 0
    for col in missing_columns:
        df[col] = 0

    # Reorder the DataFrame columns based on desired columns
    df = df[desired_columns]
    return df


def create_data_gene_image():
    result = []
    print('finding top max var genes.....')
    top_max_var_gene = find_top_gene(N=100)
    print('find top max var gene done!')
    cnts_directory = 'data/ST-cnts/'
    files = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for patient in files:
        if patient in ('A', 'B', 'C', 'D'):
            for i in range(1, 7):
                cnts_file = f'{cnts_directory}{patient}{i}.tsv/ut_{patient}{i}_stdata_filtered.tsv'
                print(cnts_file)
                #             df = pd.read_csv(cnts_file, delimiter='\t', usecols = top_max_var_gene)
                df = extract_top_gene(cnts_file, top_max_var_gene)
                for j, row in tqdm(df.iterrows()):
                    index = row['Unnamed: 0']
                    index = index.split('x')
                    x = int(index[0])
                    y = int(index[1])
                    row_value = df.iloc[j, : -1].tolist()
                    image_path = f'ST-imgs-split/{patient}/{patient}{i}/sub_img_{x}_{y}.jpg'
                    result.append((row_value, image_path, patient))

        else:
            for i in range(1, 4):
                cnts_file = cnts_directory + f"{patient}{i}.tsv/" + f'ut_{patient}{i}_stdata_filtered.tsv'
                print(cnts_file)
                #             df = pd.read_csv(cnts_file, delimiter='\t', usecols = top_max_var_gene)
                df = extract_top_gene(cnts_file, top_max_var_gene)
                for j, row in tqdm(df.iterrows()):
                    index = row['Unnamed: 0']
                    index = index.split('x')
                    x = int(index[0])
                    y = int(index[1])
                    row_value = df.iloc[j, : -1].tolist()
                    image_path = f'ST-imgs-split/{patient}/{patient}{i}/sub_img_{x}_{y}.jpg'
                    result.append((row_value, image_path, patient))
    result = pd.DataFrame(result, columns=['gene_expression', 'image_path', 'patient'])
    result = result.fillna(0)
    result = result.loc[(result != 0).any(axis=1)]
    result.to_csv('data.csv')


if __name__ == '__main__':
    # split image, resize to 256 * 256 and save to ST-imags-split
    # main_split_image()

    # create dataset: gene_expression, sub_image, patient
    create_data_gene_image()

