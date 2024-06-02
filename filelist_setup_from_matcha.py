def setup(in_path, out_path):
    lines = []
    with open(in_path, "r") as file:
        for line in file:
            line = line.split("|")[0]
            lines.append(line)
    with open(out_path, "w") as file:
        for line in lines:
            file.write(line + "\n")

train_in_path = "./multilingual_train_filelist.txt"
test_in_path = "./multilingual_test_filelist.txt"
val_in_path = "./multilingual_val_filelist.txt"
train_out_path = "./multilingual_filelist.train"
test_out_path = "./multilingual_filelist.test"
val_out_path = "./multilingual_filelist.val"

setup(train_in_path, train_out_path)
setup(test_in_path, test_out_path)
setup(val_in_path, val_out_path)