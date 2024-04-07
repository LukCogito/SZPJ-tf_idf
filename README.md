# SZPJ-tf_idf
## Information Retrieval System using TF-IDF and Cosine Similarity

### To do:
- ~~load data from docs (n=3204)~~
	+ ~~there will be dictrionary with doc ID as key and list of keywords in doc as value~~
	+ ~~keywords will be tokenized and lematized using Spacey library~~
- ~~load data from querries~~
	+ ~~the same rules as for data for docs~~
- ~~do the tf-idf magic~~

### Overview
This Python script demonstrates an information retrieval system that utilizes the TF-IDF vectorization technique and cosine similarity to match queries with relevant documents. The system takes a collection of documents and queries as input and outputs a list of the most similar documents for each query.

### Dependencies
- The script relies on the following libraries:
	+ os: Used for interacting with the file system.
	+ re: Provides regular expression matching operations.
	+ spacy: A library for advanced natural language processing.
	+ BeautifulSoup: Used for parsing HTML and XML documents.
	+ sklearn.feature_extraction.text: Provides the TF-IDF vectorizer.
	+ sklearn.metrics.pairwise: Offers cosine similarity calculation.


### Code Breakdown
- **Model Loading and Initialization**: The script attempts to load the "en_core_web_sm" model from spaCy for NLP tasks. If the model is not found, it installs it and then loads it.
- **Function Definitions**: The script defines several functions for different tasks:
	+ `get_string_from_html(file_path)`: Extracts text content from an HTML file.
	+ `remove_tail(text)`: Removes unnecessary text tails from documents.
	+ `preprocess_text(text)`: Preprocesses text by tokenization, lemmatization, and removing stop words.
	+ `get_data_from_xml(file_path)`: Extracts query text and numbers from an XML file.
	+ `do_tf_idf_magic(queries, docs, file_path)`: Performs TF-IDF vectorization and cosine similarity calculations.
	+ `transform_data(data_dict)`: Transforms dictionary data into a format suitable for TF-IDF vectorization.
- **Data Processing**: The script initializes dictionaries for storing document and query data. It iterates over a directory of HTML documents, extracting and preprocessing their content. Similarly, it processes an XML file containing queries.
- **TF-IDF and Cosine Similarity**: The `do_tf_idf_magic` function applies TF-IDF vectorization to the preprocessed queries and documents. It then computes the cosine similarity between each query and all documents. The results are sorted and stored in a text file.

### Usage
To use the script, place your HTML documents in the "documents" folder and provide an XML file containing queries in the same directory as the script. Run the script, and it will output a file named "output.txt" containing the top 100 most similar documents for each query.

### Conclusion
This script showcases a basic information retrieval system using TF-IDF and cosine similarity. It can be extended and customized to handle larger datasets and incorporate additional features for improved retrieval performance. Mean Average Precision of the system is 0.307 at the moment. That is not good nor terrible - it's just acceptable.