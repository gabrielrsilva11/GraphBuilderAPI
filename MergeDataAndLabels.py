path_data_file = "Data_to_process/Task1/Train/subtask1_train.data.txt"
path_annotations_file = "Data_to_process/Task1/Train/subtask1_train.labels.txt"
merged_file = open("Data_to_process/Task1/Merged/subtask1_merged.data.txt", "w")


with open(path_data_file) as data_file, open(path_annotations_file) as anno_file:
    for data, label in zip(data_file, anno_file):
        data_split = data.replace("\n", "").split(" ")
        label_split = label.replace("\n", "").split(" ")
        for i in range(0, len(data_split)):
            merged_file.write("{}|{} ".format(data_split[i], label_split[i]))
        merged_file.write("\n")