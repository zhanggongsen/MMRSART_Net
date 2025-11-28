import os
import shutil
Root = r"H:"
save_path=os.path.join(Root,"")

# Processed_4DCT_root=os.path.join(Root,'0_50_Preprocessed')
file_list=[]
for root, dirs, files in os.walk(Root):
    for file_i in files:
        files_path = os.path.join(root, file_i)
        if "stacked_img_MTL_" in files_path:
            file_list.append(files_path)

if not os.path.exists(save_path):
    os.mkdir(save_path)
for path_toProcess in file_list:
    print(path_toProcess)
    file_name=os.path.split(path_toProcess)[1]
    write_path=os.path.join(save_path,file_name)
    shutil.move(path_toProcess,write_path)