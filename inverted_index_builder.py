import os
from lxml import etree
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import json
from nltk.stem import PorterStemmer

"""
Builds and saves the inverted index.
The inverted index is a dictionary of the following form:
    1. Keys - All words (excluding stop words) in our corpus
    2. Values - Array of tuples of the form (record_id,tf-idf score)
"""


def build_inverted_index(path):
    inverted_index = {}
    xml_files = get_xml_files_data(path)
    data = extract_data_from_files(xml_files)
    amount_of_documents = len(data)
    df = load_data_to_dataframe(data)
    ps = PorterStemmer()
    analyzer = CountVectorizer(stop_words="english").build_analyzer()

    def stemmed_words(doc):
        return (ps.stem(w) for w in analyzer(doc))

    # Tokenize and count amount of words in each document
    cv = CountVectorizer(analyzer=stemmed_words)
    cv_matrix = cv.fit_transform(df['text'])
    vocabulary = cv.get_feature_names()
    for word in vocabulary:
        inverted_index[word] = []
    df_dtm = pd.DataFrame(cv_matrix.toarray(), index=df['record_num'].values, columns=vocabulary)

    # Builds the inverted index, initially with normalized tf scores
    for index, row in df_dtm.iterrows():
        max_frequency = max(row)
        for word in vocabulary:
            if row[word] > 0:
                inverted_index[word].append((index, (row[word]) / max_frequency))

    # Calculates the idf score for each word, and sets the tf-idf score for each word, per each document
    for word in vocabulary:
        idf_score = np.log2(amount_of_documents / len(inverted_index[word]))
        new_scores = []
        for scores in inverted_index[word]:
            tf_idf_score = scores[1] * idf_score
            new_scores.append((scores[0], tf_idf_score))
        inverted_index[word] = new_scores

    # Calculate documents length (norms) for CosSim
    docs_norm = {}
    for index, row in df_dtm.iterrows():
        docs_norm[index] = 0
        for word in vocabulary:
            if row[word] > 0:
                for doc_x_weight in inverted_index[word]:
                    if doc_x_weight[0] == index:
                        docs_norm[index] += doc_x_weight[1] * doc_x_weight[1]
                        break
                docs_norm[index] = np.sqrt(docs_norm[index])

    for word in inverted_index:
        new_scores = []
        scores = inverted_index[word]
        for score in scores:
            doc = score[0]
            doc_norm = docs_norm[doc]
            word_weight = score[1]
            new_scores.append(((doc, doc_norm), word_weight))
        inverted_index[word] = new_scores

    # Saves the inverted index
    save_index(inverted_index)


"""
Given the loaded xml files, returns a list of dictionary, containing:
    1. record_num - Unique ID of each record
    2. title - title of each document
    3. text - A textual summary of the document. Each document either has a "abstract" or "extract" part,
              Containing the textual summary.
"""


def extract_data_from_files(files):
    documents_identifier = []
    i = 0
    for file in files:
        for record_num in file.xpath("//root//RECORD//RECORDNUM//text()"):
            documents_identifier.append({"record_num": record_num})
    for file in files:
        for title in file.xpath("//root/RECORD//TITLE//text()"):
            documents_identifier[i]["title"] = title
            i += 1
    i = 0
    for file in files:
        for text in file.xpath("//root//RECORD//*[self::ABSTRACT or self::EXTRACT]//text()"):
            documents_identifier[i]["text"] = text
            i += 1

    return documents_identifier


"""
Loads the xml files in the given path, and returns them is a list.
Each item in the list can be iterated using xpath, so we can load the files one time and use it in various functions
"""


def get_xml_files_data(path):
    all_files = []
    for file in os.listdir(path):
        xml_file = open(path + '/' + file, 'r')
        data = etree.fromstring(xml_file.read())
        all_files.append(data)
    return all_files


"""
Loads the initial data dictionary to a dataframe.
Indices of the dataframe are:
    1. record_num - Unique ID of each record
    2. text - title + summary of the paper
"""


def load_data_to_dataframe(data):
    paper_nums = []
    texts = []
    for i in range(len(data)):
        paper_nums.append(data[i]["record_num"])
        texts.append(data[i]["title"] + data[i]["text"])
    df = pd.DataFrame({"record_num": paper_nums, "text": texts})
    return df


"""
Saves the inverted index to disk.
"""


def save_index(inverted_index):
    with open('vsm_inverted_index.json', 'w') as outfile:
        json.dump(inverted_index, outfile)
