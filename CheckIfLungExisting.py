import os
root=r"H:"
IDs=os.listdir(root)

for i in IDs:
    path_0=os.path.join(root,i)
    phase0_path=os.path.join(path_0,'0')
    phase50_path = os.path.join(path_0, '50')
    size_0=os.path.getsize(phase0_path)
    size_50 = os.path.getsize(phase50_path)
    # print(size_0)
    # print(size_50)

    # if size_0!=size_50 or size_0==0 or size_50==0:
    #     print("******", i, "******")
    #     print(size_0)
    #     print(size_50)
    # if_RS_exits_0=0
    # for roots, dirs, files in os.walk(phase0_path):
    #     for file in files:
    #         if 'RS' in file:
    #             if_RS_exits_0=1
    # if_RS_exits_50 = 0
    # for roots, dirs, files in os.walk(phase50_path):
    #     for file in files:
    #         if 'RS' in file:
    #             if_RS_exits_50=1
    # if if_RS_exits_0==0 or if_RS_exits_50==0:
    #     print("******", i, "******")
    #     print('RS not existing')

    phase_struct_0_path = os.path.join(path_0, 'structs_0')
    phase50_struct_50_path = os.path.join(path_0, 'structs_50')
    if_RS_exits_0=0
    for roots, dirs, files in os.walk(phase_struct_0_path):
        for file in files:
            if 'Lung' in file:
                if_RS_exits_0=1
    if_RS_exits_50 = 0
    for roots, dirs, files in os.walk(phase50_struct_50_path):
        for file in files:
            if 'Lung' in file:
                if_RS_exits_50=1
    if if_RS_exits_0==0 or if_RS_exits_50==0:
        print("******", i, "******")
        print('Lung not existing')

