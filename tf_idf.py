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
    # Close the opened file
    file.close()
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
            if lemmatized_token not in preprocessed_tokens:
                preprocessed_tokens.append(lemmatized_token)

    return preprocessed_tokens

# Define a funciton for extracting querry text and number from xml file
def get_data_from_xml(file_path):
    # Read the xml file
    with open('query_devel.xml', 'r') as file:
        xml_data = file.read()
        # Initialize list for xml docs
        xml_docs = []

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
        for doc in xml_docs:
            # Make soup out of it
            soup = BeautifulSoup(doc, "xml")
            # Extract query number
            doc_number = soup.find("DOCNO").text
            doc_number = int(doc_number)
            # Extract query text
            doc_text = soup.find("DOC").text.strip()
            # Make tokens out of the text
            doc_tokens = preprocess_text(doc_text)
            # And fill the dict
            queries[doc_number] = doc_tokens
    # Cloese the file
    file.close()
    # And return result dict with queries
    return queries

def do_tf_idf_magic(queries, docs, file_path):
    # Define a function for transforming the data to tf-idf vectorizer-friendly format
    def transform_data(data_dict):
        data_trans = []

        for value in data_dict.values():
            value_trans = " ".join(value)
            data_trans.append(value_trans)

        return data_trans

    queries_trans = transform_data(queries)
    docs_trans = transform_data(docs)
        

    # Create an instance of TfidfVectorizer
    tfidf = TfidfVectorizer()

    # Vectorize docs and queries
    sparse_doc_term_matrix = tfidf.fit_transform(docs_trans)
    sparse_query_term_matrix = tfidf.transform(queries_trans)

    # Compute cos similarity
    sim_matrix = cosine_similarity(sparse_doc_term_matrix, sparse_query_term_matrix)
    print(sim_matrix)


    query_scores = []
    keys = list(docs.keys())
    
    for i in range(sim_matrix.shape[1]):
        similarities = sim_matrix[:,i]
        dict_id_sim = {}
        for j in range(len(docs)):
            dict_id_sim[keys[j]] = similarities[j]
        query_scores.append(dict_id_sim)

    for i in range(len(query_scores)):
        query_scores[i] = dict(sorted(query_scores[i].items(), key=lambda item: item[1], reverse=True))
    
    out_text = ""
    for i, query_sims in enumerate(query_scores, 1):
        for docID, sim in list(query_sims.items())[:100]:
           out_text += f"{i}\t{docID}\t{sim}\n"

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

print(documents)

path = "./query_devel.xml"
queries = get_data_from_xml(path)
print(queries)