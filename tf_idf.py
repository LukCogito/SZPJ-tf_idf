# A script for loading data from docs, queries and tf_idf realization with ouput for each query as 100 most relevant docs
# Author:
#  _          _       ____            _ _        
# | |   _   _| | __  / ___|___   __ _(_) |_ ___  
# | |  | | | | |/ / | |   / _ \ / _` | | __/ _ \ 
# | |__| |_| |   <  | |__| (_) | (_| | | || (_) |
# |_____\__,_|_|\_\  \____\___/ \__, |_|\__\___/ 
#                               |___/            

import os
import re
import spacy
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to load en_core_web_sm model for nlp
try:
    nlp = spacy.load("en_core_web_sm")
# If it fails, just intall it and chill
except OSError:
    print("Model 'en_core_web_sm' is not installed; installing...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Define a function for extracting text from a html file
def get_string_from_html(file_path):
    # Open and read the file
    with open(file_path, "r", encoding="UTF8") as file:
        # Parse the text from html code
        soup = BeautifulSoup(file, 'html.parser')
        text = soup.get_text()
    # Return the text as string var
    return text

# Define a function for removing unnecesary tail from docs (text like "CA581203 JB March 22, 1978  8:28 PM")
def remove_tail(text):
    # Use re.search() for finding first apperance of CA and digits code in text
    keyword_pattern = re.compile(r'CA\d{3,}')
    keyword_match = re.search(keyword_pattern, text)
    
    # If it was found
    if keyword_match:
        # Return the part of string before the apperance
        return text[:keyword_match.start()]
    else:
        # If it wasn't give me the original string and let it go
        return text

# Define a function for text preprocesing
def preprocess_text(text):
    # I use spaCy
    doc = nlp(text)
    # Initialize empty list for processed tokens
    preprocessed_tokens = []

    # Iterrate one each token in doc
    for token in doc:
        # If it isn't stop word (a, and, the, etc.) and it is alpha (not a numerical value, special character, etc.)
        if not token.is_stop and token.is_alpha:
            # Lemmatize it
            lemmatized_token = str(token.lemma_).lower()
            # And append it to preprocessed tokens
            preprocessed_tokens.append(lemmatized_token)

    return preprocessed_tokens

# Define a funciton for extracting querry text and number from xml file
def get_data_from_xml(file_path):
    # Read the xml file
    with open('query_devel.xml', 'r') as file:
        xml_data = file.read()
        # Initialize a list for xml docs
        xml_docs = []

        # Initialize a list for origin doc numbers
        doc_numbers_origin = []

        # Prepare re pattern for extraction of each <DOC>content</DOC> in file
        pattern = re.compile(r'<DOC>(.*?)</DOC>', re.DOTALL)
        matches = pattern.finditer(xml_data)

        # Extract every single query
        for match in matches:
            xml_doc = match.group(0).strip()
            xml_docs.append(xml_doc)

        # Prepare dic for preprocessed queries
        queries = {}

        # Iterrate thorugh the non-processed docs
        for index, doc in enumerate(xml_docs):
            # Make soup out of it
            soup = BeautifulSoup(doc, "xml")
            # Extract query number
            doc_number = soup.find("DOCNO").text
            doc_number = int(doc_number)
            # Append origin query number to a list
            doc_numbers_origin.append(doc_number)
            # Extract query text
            doc_text = soup.find("DOC").text.strip()
            # Make tokens out of the text
            doc_tokens = preprocess_text(doc_text)
            # And fill the dict
            queries[index] = doc_tokens
    # And return result dict with queries
    return queries, doc_numbers_origin

# Define a function for the magic with the tf-idf
def do_tf_idf_magic(queries, docnos, docs, file_path):
    # Define a function for transforming the data to tf-idf vectorizer-friendly format
    def transform_data(data_dict):
        # Prepare a list
        data_trans = []

        # Iterrate over the values
        for value in data_dict.values():
            # Join value with " " as separator
            value_trans = " ".join(value)
            # And append it to the destination list
            data_trans.append(value_trans)

        return data_trans

    # Transform queries
    queries_trans = transform_data(queries)
    # Transform documents
    docs_trans = transform_data(docs)

    # Create an instance of TfidfVectorizer
    tfidf = TfidfVectorizer()

    # Vectorize docs and queries
    sparse_doc_term_matrix = tfidf.fit_transform(docs_trans)
    sparse_query_term_matrix = tfidf.transform(queries_trans)

    # Compute cos similarity
    sim_matrix = cosine_similarity(sparse_doc_term_matrix, sparse_query_term_matrix)
    
    # Convert doc keys to list (because it's more convinient for iterration)
    keys = list(docs.keys())

    # Initialize DaraFrame where columns represent queries and rows represent docs
    df = pd.DataFrame(sim_matrix, index=keys, columns=docnos)

    # Prepare list for scores
    query_scores = []

    # Iterrate over the columns
    for col in df:
        # Sort the column in descending order
        df_col_sorted = df[col].sort_values(ascending=False) # Type is series object
        # Filter the first 100 items
        df_col_sorted = df_col_sorted[:100]
        # And append col to the query scores list
        query_scores.append(df_col_sorted)

    # Sort the query scores by the values - docno
    def get_key(dict):
        return dict.name
    query_scores.sort(key=get_key)
    
    # And output the text file
    out_text = ""
    # Iterrate over the query scores list
    for query in query_scores:
        # For evry value of series object = doc IDs and similairity
        for doc_id, sim in query.items():
            # Object series has its atribute - name; in our case it is doc number (= number of column)
            docno = query.name
            # Add coresponding line to the out text
            out_text += f"{docno}\t{doc_id}\t{sim}\n"
    # Finally export
    with open(file_path, "w", encoding="UTF8") as file:
        file.write(out_text)


# Initilaize dictionary for storing doc id as a key and doc keywoards list as a value
documents = {}

# Initialize string for a path to the directory with docs
directory = "./documents/"

# Iterrate over the folder with documents
for filename in os.listdir(directory):
    # Make the script fool-proof and ignore possible different files in ./documents/
    if filename.endswith(".html"):
        # Initialize the path to the doc file
        path = directory + filename
        # Extract text from file
        text = get_string_from_html(path)
        # Remove unnecessary tail from text
        text = remove_tail(text)
        # Tokennize and lemmatize text
        tokens = preprocess_text(text)
        # Split the filename and extension
        filename_splitted = os.path.splitext(filename)
        # Assign filename without extention to the doc_id var
        doc_id = filename_splitted[0]
        # And add it as a key to the documents dict
        documents[doc_id] = tokens

path = "./query_devel.xml"
queries, docnos = get_data_from_xml(path)
do_tf_idf_magic(queries, docnos, documents, "./output.txt")