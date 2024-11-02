import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
books = pd.read_csv("Books.csv")
ratings = pd.read_csv("Ratings.csv")
books_data = books.merge(ratings, on="ISBN")

# Preprocessing
df = books_data.copy()
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df.drop(columns=["ISBN", "Year-Of-Publication", "Image-URL-S", "Image-URL-M"], axis=1, inplace=True)
df.drop(index=df[df["Book-Rating"] == 0].index, inplace=True)
df["Book-Title"] = df["Book-Title"].apply(lambda x: re.sub("[\W_]+", " ", x).strip())

# Function to get popular books
def popular_books(df, n=100):
    rating_count = df.groupby("Book-Title").count()["Book-Rating"].reset_index()
    rating_count.rename(columns={"Book-Rating": "NumberOfVotes"}, inplace=True)

    rating_average = df.groupby("Book-Title")["Book-Rating"].mean().reset_index()
    rating_average.rename(columns={"Book-Rating": "AverageRatings"}, inplace=True)

    popularBooks = rating_count.merge(rating_average, on="Book-Title")

    def weighted_rate(x):
        v = x["NumberOfVotes"]
        R = x["AverageRatings"]
        return ((v * R) + (m * C)) / (v + m)

    C = popularBooks["AverageRatings"].mean()
    m = popularBooks["NumberOfVotes"].quantile(0.90)

    popularBooks = popularBooks[popularBooks["NumberOfVotes"] >= 250]
    popularBooks["Popularity"] = popularBooks.apply(weighted_rate, axis=1)
    popularBooks = popularBooks.sort_values(by="Popularity", ascending=False)
    return popularBooks[["Book-Title", "NumberOfVotes", "AverageRatings", "Popularity"]].reset_index(drop=True).head(n)

# Display top popular books
top_ten = popular_books(df, 5)
print("📘 MOST POPULAR 5 BOOKS")
print(top_ten)

# Select a book for recommendations
book_names = df["Book-Title"].value_counts().index[:200]
print("\nChoose the book you're reading for advice:")
for i, name in enumerate(book_names):
    print(f"{i + 1}: {name}")

# Get user input for selected book
selected_index = int(input("\nEnter the number of the book you choose: ")) - 1
book = book_names[selected_index]

# Item-based recommendations
def item_based(bookTitle):
    bookTitle = str(bookTitle)
    img_list_item = []

    if bookTitle in df["Book-Title"].values:
        rating_count = pd.DataFrame(df["Book-Title"].value_counts())
        rare_books = rating_count[rating_count["Book-Title"] <= 200].index
        common_books = df[~df["Book-Title"].isin(rare_books)]

        if bookTitle in rare_books:
            print("No Recommendations for this Book ☹️")
        else:
            common_books_pivot = common_books.pivot_table(index=["User-ID"], columns=["Book-Title"], values="Book-Rating")
            title = common_books_pivot[bookTitle]
            recommendation_df = pd.DataFrame(common_books_pivot.corrwith(title).sort_values(ascending=False)).reset_index(drop=False)

            recommendation_df = recommendation_df[recommendation_df["Book-Title"] != bookTitle]
            recommendation_df = recommendation_df.head(5)
            recommendation_df.columns = ["Book-Title", "Correlation"]

            print("\n📗 TOP RATED BOOKS")
            for i in range(len(recommendation_df["Book-Title"].tolist())):
                img_url = df.loc[df["Book-Title"] == recommendation_df["Book-Title"].tolist()[i], "Image-URL-L"][:1].values[0]
                img_list_item.append(img_url)
                print(recommendation_df["Book-Title"].tolist()[i], img_url)

item_based(book)

# Content-based recommendations
def content_based(bookTitle):
    bookTitle = str(bookTitle)
    img_list_content = []

    if bookTitle in df["Book-Title"].values:
        rating_count = pd.DataFrame(df["Book-Title"].value_counts())
        rare_books = rating_count[rating_count["Book-Title"] <= 200].index
        common_books = df[~df["Book-Title"].isin(rare_books)]

        if bookTitle in rare_books:
            print("No Recommendations for this Book ☹️")
        else:
            common_books = common_books.drop_duplicates(subset=["Book-Title"])
            common_books.reset_index(inplace=True)
            common_books["index"] = [i for i in range(common_books.shape[0])]
            targets = ["Book-Title", "Book-Author", "Publisher"]
            common_books["all_features"] = [" ".join(common_books[targets].iloc[i,].values) for i in range(common_books[targets].shape[0])]
            vectorizer = CountVectorizer()
            common_booksVector = vectorizer.fit_transform(common_books["all_features"])
            similarity = cosine_similarity(common_booksVector)
            index = common_books[common_books["Book-Title"] == bookTitle]["index"].values[0]
            similar_books = list(enumerate(similarity[index]))
            similar_booksSorted = sorted(similar_books, key=lambda x: x[1], reverse=True)[1:6]
            books = [common_books[common_books["index"] == similar_booksSorted[i][0]]["Book-Title"].item() for i in range(len(similar_booksSorted))]

            print("\n📕 OTHER USERS' SELECTIONS")
            for book in books:
                img_url = common_books.loc[common_books["Book-Title"] == book, "Image-URL-L"][:1].values[0]
                img_list_content.append(img_url)
                print(book, img_url)

content_based(book)
