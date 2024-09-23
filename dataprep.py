import cv2
import numpy as np
import os, re
import pandas as pd

class DataPreparation:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    # save filepath
    def link_recursive(self, path):
        df = pd.DataFrame()

        directories_path = os.listdir(path)

        for root, directories, files in os.walk(path):
            for file in files:
                df_loop = pd.DataFrame()
                df_loop['link'] = [f'{root}\{file}']
                df_loop['filetipe'] = df_loop['link'].apply(lambda x: 'img' if not x.endswith('txt') else 'txt')
                df_loop['basename'] = df_loop['link'].apply(lambda x: re.sub(r'\..*$', '', os.path.basename(x)))
                df_loop['class'] = df_loop['link'].apply(lambda x: rf'{x}'.split("\\")[2])


                df = pd.concat([df, df_loop], axis=0)


        df.to_csv('link_prepare.csv' ,index=False)

        return df


    # read bbox for polygon
    def polygon_to_bounding_box(self, polygon, image_width, image_height):
        # Split the polygon string into a list of floats
        coords = list(map(float, polygon.split()))

        # Extract x and y coordinates
        x_coords = coords[0::2]  # All x coordinates
        y_coords = coords[1::2]  # All y coordinates

        # Find the min and max x and y to get the bounding box
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Convert normalized coordinates to pixel coordinates
        bounding_box = (
            int(min_x * image_width),  # left
            int(min_y * image_height), # top
            int(max_x * image_width),  # right
            int(max_y * image_height)   # bottom
        )

        return bounding_box 

    # crop images with images and annotations
    def crop_images(self, image_path, annotations):
        try:
            image = cv2.imread(image_path)

            h, w, _ = image.shape
            cropped_images = []

            for polygon in annotations:
                class_id = polygon[0]
                x1, y1, x2, y2 = self.polygon_to_bounding_box(polygon[2:], w, h)

                # Crop the image
                cropped_image = image[y1:y2, x1:x2]
                cropped_images.append((class_id, cropped_image))

            return cropped_images
        except:
            raise FileNotFoundError(f"Image not found: {image_path}")

    # save result of images cropped
    def save_cropped_images(self, cropped_images, category):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        directory = os.path.join(self.output_dir, category)
        if not os.path.exists(directory):
            os.makedirs(f'{self.output_dir}/{category}')

        for class_id, img in cropped_images:
            # Save the cropped image with a unique filename
            filename = os.path.join(f'{self.output_dir}/{category}', f"cropped_class_{int(class_id)}_{np.random.randint(0, 10000)}.jpg")
            cv2.imwrite(filename, img)

    # read annotation file
    def read_annotation(self, annotation_file):
        with open(annotation_file, 'r') as f:
            annotations = [line.rstrip() for line in f]

        return annotations