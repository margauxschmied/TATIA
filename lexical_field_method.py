import pandas as pd
from main import clean_text
from fuzzywuzzy import fuzz

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

    genre=""
    note=0
    clean_test = clean_text(text)
    for i in range(len(df)):
        tmp=similar(clean_test, lexical[i])
        if note < tmp:
            note = tmp
            genre=df["genre"][i]

    print(genre)
    return genre


# not working version
def get_similarities_fuzzy(df, text):
    lexical = df["lexical_field"]

    genre=""
    note=0
    clean_test = clean_text(text)
    for i in range(len(df)):
        tmp=fuzz.ratio(clean_test, lexical[i])
        if note < tmp:
            note = tmp
            genre=df["genre"][i]

    print(genre)


df = pd.read_csv("archive/lexical_field_clean.csv")

# clean_lexical(df)
get_similarities_counter(df, "Twenty-five years after the original series of murders in Woodsboro, a new killer emerges, and Sidney Prescott must return to uncover the truth.")
#get_similarities_fuzzy(df, "Twenty-five years after the original series of murders in Woodsboro, a new killer emerges, and Sidney Prescott must return to uncover the truth.")

#fuzz.token_sort_ratio("Twenty-five years after the original series of murders in Woodsboro, a new killer emerges, and Sidney Prescott must return to uncover the truth.", s4)