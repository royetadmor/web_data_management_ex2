import json
import numpy as np

"""
Given a path and a question:
    1. Loads invereted index from disk
    2. Extracts all relevant documents. That is, documents containing at least one word from the question
    3. Extracts a list of records_nums of relevant documents
    4. Builds the matrix as showen in lecture. the i,j cell is the tf-idf score of word j in document i.
    For easier use, it's kept in a dictionary.
    5. Calculates the query vector. each word gets a tf-idf score, based on:
        5.1 Non-normalized term frequency with respect to the query
        5.2 IDF score, as showen in lectures
    6. Calculates the cosine similarity for each document
    7. Sorts the result and keeps only the documents that the similarity is greater than 0.5
    8. Why 0.5 you ask? no clue, it's just what looks best when comparing to the actual results. 
"""


def query(path, question):
    res = []
    inverted_index = load_inverted_index(path)
    relevant_index, relevant_words = get_all_relevant_documents(question, inverted_index)
    relevant_record_num = get_record_nums(relevant_index)
    record_dict = get_document_matrix(relevant_index, relevant_words, relevant_record_num)
    query_vector = calculate_query_vector(inverted_index, relevant_words)
    for record in record_dict:
        similarity = calculate_similarity(query_vector, record_dict[record])
        res.append((record, similarity))
    res = (sorted(res, key=lambda tup: tup[1], reverse=True))
    relevant_res = []
    for data in res:
        if (data[1] > 0.5):
            relevant_res.append(data[0])
    print(relevant_res)
    return relevant_res


"""
Given a question and the inverted index, returns:
    1. A dictionary containing:
        1.1 Key: words in the question
        1.2 Value: their tf-idf score
    2. A list of words that are both in the index and in the question
"""


def get_all_relevant_documents(question, inverted_index):
    res = {}
    relevant_words = []
    words = question.split(' ')
    for word in words:
        word = word.strip('?').lower()
        if (word in inverted_index):
            relevant_words.append(word)
            res[word] = inverted_index[word]
    return res, relevant_words


"""
Loads and returns the inverted index from the path
"""


def load_inverted_index(path):
    with open(path) as f:
        data = json.load(f)
        return data


"""
Extracts and returns all record_nums from the inverted index
"""


def get_record_nums(inverted_index):
    relevant_records = []
    for word in inverted_index:
        for data in inverted_index[word]:
            relevant_records.append(data[0])
    return relevant_records


"""
Given the relevant part of the inverted index, the relevant words and the record_nums, returns:
    1. A dictionary containing:
        1.1 Key: Record num
        1.2 Value: vector of len(relevant_words) entries. Each entry is the tf-idf score of the j'th word.
"""


def get_document_matrix(relevant_index, relevant_words, relevant_record_num):
    record_dict = {}
    for record_num in relevant_record_num:
        record_dict[record_num] = [0] * len(relevant_words)
    for i in range(len(relevant_words)):
        for docs in relevant_index[relevant_words[i]]:
            record_dict[docs[0]][i] = docs[1]

    return record_dict


"""
Returns the amount of distinct words in the inverted index
"""


def get_amount_of_records(inverted_index):
    res = set()
    for word in inverted_index:
        for data in inverted_index[word]:
            res.add(data[0])
    return len(res)


"""
Calculates the query vector.
For each word, calculate it's tf-idf score and appends it to the vector.
TODO: I assumed the term frequency will be 1, but it's not always true. 
We should calculate the term frequency for each term.
"""


def calculate_query_vector(inverted_index, relevant_words):
    query_vector = []
    amount_of_records = get_amount_of_records(inverted_index)
    for word in relevant_words:
        relevant_docs_amount = len(inverted_index[word])
        score = np.log(amount_of_records / relevant_docs_amount)
        query_vector.append(score)
    return query_vector


"""
Returns the cosine similarity between the query vector and a given vector.
"""


def calculate_similarity(query_vector, record_vector):
    return np.dot(query_vector, record_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(record_vector))


query("vsm_inverted_index.json",
      "How are salivary glycoproteins from CF patients different from those of normal subjects?")
