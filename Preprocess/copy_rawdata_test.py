import os
import shutil

def copy_selected_subfolders(src_root, dst_root):

    os.makedirs(dst_root, exist_ok=True)

    for subfolder_name in os.listdir(src_root):
        subfolder_path = os.path.join(src_root, subfolder_name)

        if os.path.isdir(subfolder_path):

            for folder_to_copy in ["structs_0", "structs_50"]:
                src_subdir = os.path.join(subfolder_path, folder_to_copy)

                if os.path.exists(src_subdir) and os.path.isdir(src_subdir):
                    dst_subdir = os.path.join(dst_root, subfolder_name, folder_to_copy)
                    os.makedirs(os.path.dirname(dst_subdir), exist_ok=True)
                    shutil.copytree(src_subdir, dst_subdir, dirs_exist_ok=True)
                    print(f"copy：{src_subdir} into {dst_subdir}")
                else:
                    print(f"skip：{src_subdir} not exist")

source_root = r"H:"
destination_root = r"C:"

copy_selected_subfolders(source_root, destination_root)
