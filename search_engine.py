import json
import numpy as np
from nltk import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

"""
Given a path and a question:
    1. Loads inverted index from disk
    2. Creates inverted index for query
    3. For each token:
        * Extracts a list of token_occurrences ((doc number, doc norm), weight)
        * Calculate CosSim (not normalized) of query and document
    4. Normalizes CosSim
    5. Sort retrieved documents by similarity score
    6. Saves (rel_res_percentage)% documents with score higher than score_threshold
"""


def retrieve_documents(path, question):
    inverted_index = load_inverted_index(path)
    query_hashmap = create_query_inverted_index(question, inverted_index)
    query = {}
    docs_x_scores = {}
    docs_x_length = {}

    for token in query_hashmap:
        idf = calc_idf(token, inverted_index)
        tf = calc_tf(token, query_hashmap)
        token_tf_idf = tf * idf
        query[token] = token_tf_idf
        token_occurrences = inverted_index[token]
        for occurrence in token_occurrences:
            doc = occurrence[0][0]
            docs_x_length[doc] = occurrence[0][1]
            doc_tf_idf = occurrence[1]
            if doc not in docs_x_scores:
                docs_x_scores[doc] = 0
            docs_x_scores[doc] += token_tf_idf * doc_tf_idf

    for doc in docs_x_scores:
        doc_length = docs_x_length[doc]
        query_length = get_norm(query)
        docs_x_scores[doc] = docs_x_scores[doc] / (doc_length * query_length)

    res = sorted(docs_x_scores.items(), key=lambda item: item[1], reverse=True)
    relevant_res = []
    score_threshold = 0.45
    rel_res_percentage = 0.8
    while len(relevant_res) == 0:
        for data in res:
            if data[1] > score_threshold:
                relevant_res.append(data[0])

        relevant_res = [r for r in relevant_res[:int(len(relevant_res) * rel_res_percentage)]]
        score_threshold -= 0.05
        rel_res_percentage -= 0.11

    with open('ranked_query_docs.txt', 'w') as f:
        for doc in relevant_res:
            f.write(doc + '\n')
    return relevant_res


"""
Creates inverted index for query.
"""


def create_query_inverted_index(question, inverted_index):
    query = {}
    ps = PorterStemmer()
    analyzer = CountVectorizer(stop_words="english").build_analyzer()

    def stemmed_words(doc):
        return (ps.stem(w) for w in analyzer(doc))

    cv = CountVectorizer(analyzer=stemmed_words)
    cv.fit_transform([question[:-1]])
    vocabulary = cv.get_feature_names()
    for word in vocabulary:
        if word in inverted_index:
            if word not in query:
                query[word] = 0
            query[word] += 1

    return query


"""
Calculates idf for token.
"""


def calc_idf(token, inverted_index):
    token_docs_length = len(inverted_index[token])
    amount_of_records = get_amount_of_records(inverted_index)
    return np.log2(amount_of_records / token_docs_length)


"""
Calculates tf for token in query.
"""


def calc_tf(token, query):
    m = query[max(query)]
    return query[token] / m


"""
Calculates query length (norm).
"""


def get_norm(query):
    norm = 0
    for token in query:
        norm += query[token] * query[token]
    norm = np.sqrt(norm)

    return norm


"""
Loads and returns the inverted index from the path
"""


def load_inverted_index(path):
    with open(path) as f:
        data = json.load(f)
        return data


"""
Returns the amount of distinct words in the inverted index
"""


def get_amount_of_records(inverted_index):
    res = set()
    for word in inverted_index:
        for data in inverted_index[word]:
            res.add(data[0][0])
    return len(res)

