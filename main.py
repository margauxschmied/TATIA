import pandas as pd


def to_csv(file, output):
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    df = pd.DataFrame(columns=["id", "title", "description"])

    for i in range(len(lines)):
        line = lines[i].split(":::")

        df_tmp = pd.DataFrame([(int(line[0]), line[1], line[2])], columns=["id", "title", "description"])
        df = df.append(df_tmp)
    df.to_csv(output)
    print(output, "Saved")

def to_csv_train_solution(file, output):
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    df = pd.DataFrame(columns=["id", "title", "genre", "description"])

    for i in range(len(lines)):
        line = lines[i].split(":::")

        df_tmp = pd.DataFrame([(int(line[0]), line[1], line[2], line[3])], columns=["id", "title", "genre", "description"])
        df = df.append(df_tmp)
    df.to_csv(output)
    print(output, "Saved")


if __name__ == "__main__":
    # to_csv("archive/dataset/test_data.txt", "test_data.csv")
    to_csv_train_solution("archive/dataset/test_data_solution.txt", "test_data_solution.csv")
