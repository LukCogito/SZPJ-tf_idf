# A script for loading data from docs and tf_idf realization
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
        # If it isn't stop word (a, and, the, etc.) and it is alpha (nut a numerical value, special character, etc.)
        if not token.is_stop and token.is_alpha:
            # Lemmatize it
            lemmatized_token = token.lemma_
            if (str(lemmatized_token) not in preprocessed_tokens) and (str(lemmatized_token).lower() not in preprocessed_tokens):
                if token.pos_ == "PROPN":
                    # Add it's lematized version
                    preprocessed_tokens.append(str(lemmatized_token))
                else:
                    # Add it's lower lemmatized version
                    preprocessed_tokens.append(str(lemmatized_token).lower())

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


# Initilaize dictionary for storing doc id as a key and doc keywoards list as a value
documents = {}

# Initialize string for a path to the directory with docs
directory = "./documents/"

# Iterrate throug folder with documents
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