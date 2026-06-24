import config
import random
import shutil
import os

def split_dataset(data_dir, train_dir, test_dir, split_ratio=0.8):
    
    for img_class in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, img_class)
        files = os.listdir(subdir_path)
        random.shuffle(files)
        num_files = len(files)
        num_train = int(split_ratio * num_files)

        train_files = files[:num_train]
        test_files = files[num_train:]

        # Train and test directories for the current class
        os.makedirs(os.path.join(train_dir, img_class), exist_ok=True)
        os.makedirs(os.path.join(test_dir, img_class), exist_ok=True)

        for f in train_files:
            src_path = os.path.join(data_dir, img_class, f)
            dst_path = os.path.join(train_dir, img_class, f)
            shutil.move(src_path, dst_path)
        
        for f in test_files:
            src_path = os.path.join(data_dir, img_class, f)
            dst_path = os.path.join(test_dir, img_class, f)
            shutil.move(src_path, dst_path)

if __name__ == "__main__":
    id_data_dir = config.id_data_path
    # Directories for train and test splits
    train_id_dir = config.train_id_data_path
    test_id_dir = config.test_id_data_path

    split_dataset(id_data_dir, train_id_dir, test_id_dir)