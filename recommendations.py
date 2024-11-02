import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read data
books = pd.read_csv("Books.csv", dtype={'ISBN': str, 'Book-Title': str, 'Book-Author': str, 'Year-Of-Publication': str, 'Publisher': str, 'Image-URL-L': str})
ratings = pd.read_csv("Ratings.csv", dtype={'User-ID': str, 'ISBN': str, 'Book-Rating': int})
books_data = books.merge(ratings, on="ISBN")

# Preprocessing
df = books_data.copy()
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df.drop(columns=["ISBN", "Year-Of-Publication", "Image-URL-S", "Image-URL-M"], axis=1, inplace=True)
df.drop(index=df[df["Book-Rating"] == 0].index, inplace=True)
df["Book-Title"] = df["Book-Title"].apply(lambda x: re.sub(r"[\W_]+", " ", x).strip())

# Streamlit setup
header = st.container()
with header:
    title = '<p style="font-family:Arial Bold; color:Purple; font-size: 80px;">📚 BOOK 📚 RECOMMENDER</p>'
    st.markdown(title, unsafe_allow_html=True)

popularity = st.container()
book_names = df["Book-Title"].value_counts().reset_index()
book_names.columns = ["names", "count"]
book = st.selectbox("Choose the book you're reading for advice:", book_names["names"])

# Content-Based Filtering
img_list_content = []

def content_based(bookTitle):
    bookTitle = str(bookTitle)
    
    if bookTitle in df["Book-Title"].values:
        rating_count = df["Book-Title"].value_counts().reset_index()
        rating_count.columns = ["Book-Title", "count"]
        rare_books = rating_count[rating_count["count"] <= 200]["Book-Title"]
        common_books = df[~df["Book-Title"].isin(rare_books)]
        
        if bookTitle in rare_books.values:
            st.warning("No Recommendations for this Book ☹️")
        else:
            common_books = common_books.drop_duplicates(subset=["Book-Title"])
            common_books.reset_index(inplace=True)
            common_books["index"] = [i for i in range(common_books.shape[0])]
            targets = ["Book-Title", "Book-Author", "Publisher"]
            common_books["all_features"] = [" ".join(common_books[targets].iloc[i].values) for i in range(common_books[targets].shape[0])]
            vectorizer = CountVectorizer()
            common_booksVector = vectorizer.fit_transform(common_books["all_features"])
            similarity = cosine_similarity(common_booksVector)
            index = common_books[common_books["Book-Title"] == bookTitle]["index"].values[0]
            similar_books = list(enumerate(similarity[index]))
            similar_booksSorted = sorted(similar_books, key=lambda x: x[1], reverse=True)[1:6]
            books = [common_books[common_books["index"] == i[0]]["Book-Title"].item() for i in similar_booksSorted]
            
            content_base = '<p style="font-family:Helvetica; color:Crimson; font-size: 40px;">📕OTHER USERS` SELECTIONS</p>'
            st.markdown(content_base, unsafe_allow_html=True)
            for book in books:
                url = common_books.loc[common_books["Book-Title"] == book, "Image-URL-L"][:1].values[0]
                img_list_content.append(url)
            st.image(img_list_content, width=130)

content_based(book)

















# import streamlit as st
# import pandas as pd
# import re
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Read data
# books = pd.read_csv("Books.csv", dtype={'Book-Title': str, 'Book-Author': str, 'Publisher': str}, low_memory=False)
# ratings = pd.read_csv("Ratings.csv")
# books_data = books.merge(ratings, on="ISBN")

# # Preprocessing
# df = books_data.copy()
# df.dropna(inplace=True)
# df.reset_index(drop=True, inplace=True)
# df.drop(columns=["ISBN", "Year-Of-Publication", "Image-URL-S", "Image-URL-M"], axis=1, inplace=True)
# df.drop(index=df[df["Book-Rating"] == 0].index, inplace=True)
# df["Book-Title"] = df["Book-Title"].apply(lambda x: re.sub(r"[\W_]+", " ", x).strip())

# # Streamlit setup
# header = st.container()
# with header:
#     title = '<p style="font-family:Arial Bold; color:Purple; font-size: 80px;">📚 BOOK 📚 RECOMMENDER</p>'
#     st.markdown(title, unsafe_allow_html=True)

# popularity = st.container()
# book_names = df["Book-Title"].value_counts().reset_index()
# book_names.columns = ["names", "count"]
# book = st.selectbox("Choose the book you're reading for advice:", book_names["names"])

# # Content-Based Filtering
# img_list_content = []

# def content_based(bookTitle):
#     bookTitle = str(bookTitle)
    
#     if bookTitle in df["Book-Title"].values:
#         rating_count = df["Book-Title"].value_counts().reset_index()
#         rating_count.columns = ["Book-Title", "count"]
#         rare_books = rating_count[rating_count["count"] <= 200]["Book-Title"]
#         common_books = df[~df["Book-Title"].isin(rare_books)]
        
#         if bookTitle in rare_books.values:
#             st.warning("No Recommendations for this Book ☹️")
#         else:
#             common_books = common_books.drop_duplicates(subset=["Book-Title"])
#             common_books.reset_index(inplace=True)
#             common_books["index"] = [i for i in range(common_books.shape[0])]
#             targets = ["Book-Title", "Book-Author", "Publisher"]
#             common_books["all_features"] = [" ".join(common_books[targets].iloc[i].values) for i in range(common_books[targets].shape[0])]
#             vectorizer = CountVectorizer()
#             common_booksVector = vectorizer.fit_transform(common_books["all_features"])
#             similarity = cosine_similarity(common_booksVector)
#             index = common_books[common_books["Book-Title"] == bookTitle]["index"].values[0]
#             similar_books = list(enumerate(similarity[index]))
#             similar_booksSorted = sorted(similar_books, key=lambda x: x[1], reverse=True)[1:6]
#             books = [common_books[common_books["index"] == i[0]]["Book-Title"].item() for i in similar_booksSorted]
            
#             content_base = '<p style="font-family:Helvetica; color:Crimson; font-size: 40px;">📕OTHER USERS` SELECTIONS</p>'
#             st.markdown(content_base, unsafe_allow_html=True)
#             for book in books:
#                 url = common_books.loc[common_books["Book-Title"] == book, "Image-URL-L"][:1].values[0]
#                 img_list_content.append(url)
#             st.image(img_list_content, width=130)

# content_based(book)
