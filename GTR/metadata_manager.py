"""
Store all dataset/task relevant meta data here for passing them to the training script.
"""


def get_meta(task):
    if task == "LIDC":
        meta = {
            "description": "LIDC Lung Module Dataset (subset with 4 annotations)",
            "channels": 1,
            # NOTE: LIDC 数据加载代码中使用 os.listdir(data_path)，
            # 所以这里必须是「目录路径」并且以 '/' 结尾，而不是具体的 .pickle 文件路径。
            # 目录下所有以 .pickle 结尾的文件都会被依次加载。
            "all_data_path": "/home/u2024111264/share/zhk/ATFM-main/LIDC/data/",
            "masking_threshold": 0.5,
            "image_size": 128,
            "admissible_size": 128,
            "output_size": 128,
            "directory_name": "LIDC",
            "raters": 4,
            "num_filters": [32, 64, 128, 192],
            # 'lossfunction': define lossfunction here
        }
        return meta

    if task == "isic3_style_concat":
        meta = {
            "description": "ISIC Skin Lesion Dataset with same split as style subsets ",
            "channels": 3,
            "all_data_path": "/home/u2024111264/share/zhk/GTRasam/GTR/data/isic256_3_style",
            "masking_threshold": 0.5,
            "image_size": 256,
            "admissible_size": 340,
            "output_size": 252,
            "directory_name": "isic3",
            "raters": 3,
            "num_filters": [32, 64, 128, 192],
            # 'lossfunction': define lossfunction here
        }
        return meta

    if task == "ISBI2016":
        meta = {
            "description": "ISBI 2016 ISIC Part 1 skin lesion segmentation dataset",
            "channels": 3,
            # 根目录下包含 ISBI2016_ISIC_Part1_Training_Data / _Training_GroundTruth 等子目录
            "all_data_path": "/home/u2024111264/share/zhk/GTRasam/ISBI",
            "masking_threshold": 0.5,
            "image_size": 256,
            "admissible_size": 340,
            "output_size": 252,
            "directory_name": "ISBI2016",
            # 实际只有 1 个标注者，这里保留字段仅作说明（在 Dataset 内部复制为 3 份以兼容原逻辑）
            "raters": 1,
            "num_filters": [32, 64, 128, 192],
            # 'lossfunction': define lossfunction here
        }
        return meta

    if task == "QUBIQ":
        meta = {
            "description": "QUBIQ multi-rater organ segmentation dataset",
            "channels": 1,
            # train/val 根目录（各 task 子目录在内部）
            "all_data_path": "/home/u2024111264/share/zhk/GTRasam/QUBIQ",
            "masking_threshold": 0.5,
            "image_size": 128,
            "admissible_size": 128,
            "output_size": 128,
            "directory_name": "QUBIQ",
            # 不同 task 标注数不同，此处仅为占位
            "raters": 2,
            "num_filters": [32, 64, 128, 192],
        }
        return meta

# brain-growth：7 个标注
# brain-tumor：3 个标注
# kidney：3 个标注
# pancreas：2 个标注
# pancreatic-lesion：2 个标注
# prostate：6 个标注
