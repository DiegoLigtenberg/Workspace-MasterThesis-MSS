import argparse


SPECTROGRAM_PATHS = {
    # IRMAS TRAIN + PART 2 OF IRMAS TEST SET
    "train":{"base":                "G:/Thesis/MIR_datasets/train_dataset/spectrogram_base",
             "no_postprocessing":   "G:/Thesis/MIR_datasets/train_dataset/spectrogram_no_post",
             "with_postprocessing": "G:/Thesis/MIR_datasets/train_dataset/spectrogram_with_post",
             "labels":              "MIR_datasets/MIR_train_labels_merged.csv"},

    # PART 1 OF IRMAS DATASET
    "val":  {"base":                "G:/Thesis/MIR_datasets/test_dataset/spectrogram_base", 
            "no_postprocessing":    "G:/Thesis/MIR_datasets/test_dataset/spectrogram_no_post",
            "with_postprocessing":  "G:/Thesis/MIR_datasets/test_dataset/spectrogram_with_post",
            "labels":               "MIR_datasets/MIR_test_labels.csv"},

    # part 3 OF IRMAS DATASET
    "test": {"base":                "G:/Thesis/MIR_datasets/test_dataset/_c_spectrogram_base", 
            "no_postprocessing":    "G:/Thesis/MIR_datasets/test_dataset/_c_spectrogram_no_post",
            "with_postprocessing":  "G:/Thesis/MIR_datasets/test_dataset/_c_spectrogram_with_post",
            "labels":               "MIR_datasets/MIR_test_labels_combined.csv"}
}


def add_data_args(parent_parser):
    pass
    




if __name__ == "__main__":
    parent_parser = argparse.ArgumentParser(description="ParentParser for MIR")
    # name, type, help
    parser = parent_parser.add_argument_group("Data")

    #  (positional arguments) must be in order
    #n ame

    # Optional arguments          
    # --long_name    (long notation)
    # -l_n          (short notation)
    parser.add_argument("--data_dirs",type=list,default=SPECTROGRAM_PATHS,
                        help="list of directory files")
    parser.add_argument("--train",type=str,default="testing")

    args=parent_parser.parse_args()
    # parent_parser.grp


    print(args.data_dirs["train"]["base"])
    print(args.train)
    # print(args)

    print(vars(args))
    print(parent_parser.print_help())



