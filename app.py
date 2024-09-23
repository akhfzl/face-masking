from utilization import read_csv, collect_crop_images


if __name__ == '__main__':
    df = read_csv('link_prepare.csv')
    saving = collect_crop_images(df)

    print("Cropped images have been saved.")