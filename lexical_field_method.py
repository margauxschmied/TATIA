import pandas as pd
from tqdm import trange

from classifier import clean_text


def clean_lexical(df):
    lexical = df["lexical_field"]
    # new_df = pd.DataFrame()
    for i in range(len(df)):
        splited = lexical[i].split(" ")
        l = clean_text(" ".join(set(splited)))
        df["lexical_field"][i] = l

    df.to_csv("archive/lexical_field_clean.csv")


def similar(summary, lexical_field):
    summary = summary.split(" ")
    lexical_field = lexical_field.split(" ")

    ret = 0
    for s in summary:
        if s in lexical_field:
            ret += 1
    return ret


# partially working version


def get_similarities_counter(df, text):
    lexical = df["lexical_field"]

    genre = ""
    note = 0
    clean_test = clean_text(text)
    for i in range(len(df)):
        tmp = similar(clean_test, lexical[i])
        if note < tmp:
            note = tmp
            genre = df["genre"][i]

    # print(genre)
    return genre


df = pd.read_csv("archive/lexical_field_clean.csv")
# clean_lexical(df)
data = pd.read_csv("archive/dataset_csv/test_data_solution.csv")

sentences = data["description"].values

genrePredit = ""
error = 0

for i in trange(len(sentences)):
    genrePredit = get_similarities_counter(df, sentences[i])
    if genrePredit != data["genre"][i]:
        error += 1
        # print(genrePredit+" "+data["genre"][i])

error = (error / len(sentences)) * 100
print(error)
