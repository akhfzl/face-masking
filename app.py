# from utilization import read_csv, collect_crop_images
from dataprep import DataPreparation


if __name__ == '__main__':
    dataprep = DataPreparation('extracted_images')
    link = dataprep.link_recursive('extracted_images')
    print(link)
    # df = read_csv('link_prepare.csv')
    # saving = collect_crop_images(df)

    print("Cropped images have been saved.")