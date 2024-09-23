import pandas as pd 
from dataprep import DataPreparation

# dataprep = DataPreparation(output_directory)
# link = dataprep.link_recursive(paths)
# print(link)
# annotations = dataprep.read_annotation(annotation_file)
# Crop the images based on the annotations
# cropped_images = dataprep.crop_images(image_file, annotations)

# Save the cropped images
# dataprep.save_cropped_images(cropped_images)

def filenames():
    output_directory = 'extracted_images'
    img_paths = 'data-cvl'

    return output_directory, img_paths

def read_csv(filepath):
    return pd.read_csv(filepath, sep=',')

def collect_crop_images(df):
    # give us the filenames
    output_directory, img_paths = filenames()

    # object
    dataprep = DataPreparation(output_directory)

    # images and labels filter
    images = df[df['filetipe'] == 'img']
    labels = df[df['filetipe'] == 'txt']

    # loop based on images len
    idx = 0
    for i in range(len(images)):
        idx += 1  

        # annotation process
        annotation_file = labels[labels['basename'] == images['basename'][i]]['link'].values[0]
        annotations = dataprep.read_annotation(annotation_file)

        # Crop the images based on the annotations
        image_file = images['link'][i]
        cropped_images = dataprep.crop_images(image_file, annotations)

        # Save the cropped images
        dataprep.save_cropped_images(cropped_images, images['class'][i])  

    
    return f'{idx} - processed'