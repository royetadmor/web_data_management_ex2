import json
import numpy as np

def query(path, question):
    res = []
    inverted_index = load_inverted_index(path)
    relevant_index, relevant_words = get_all_relevant_documents(question, inverted_index)
    relevant_record_num = get_record_nums(relevant_index)
    record_dict = get_document_matrix(relevant_index, relevant_words, relevant_record_num)
    query_vector = calculate_query_vector(inverted_index,relevant_words)
    for record in record_dict:
        similarity = calculate_similarity(query_vector,record_dict[record])
        res.append((record,similarity))
    res = (sorted(res, key=lambda tup: tup[1],reverse=True))
    relevant_res = []
    for data in res:
        if(data[1] > 0.5):
            relevant_res.append(data[0])
    print(relevant_res)
    return relevant_res
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


def load_inverted_index(path):
    with open(path) as f:
        data = json.load(f)
        return data


def get_record_nums(inverted_index):
    relevant_records = []
    for word in inverted_index:
        for data in inverted_index[word]:
            relevant_records.append(data[0])
    return relevant_records


def get_document_matrix(relevant_index, relevant_words, relevant_record_num):
    record_dict = {}
    for record_num in relevant_record_num:
        record_dict[record_num] = [0] * len(relevant_words)
    for i in range(len(relevant_words)):
        for docs in relevant_index[relevant_words[i]]:
            record_dict[docs[0]][i] = docs[1]

    return record_dict


def get_amount_of_records(inverted_index):
    res = set()
    for word in inverted_index:
        for data in inverted_index[word]:
            res.add(data[0])
    return len(res)
def calculate_query_vector(inverted_index,relevant_words):
    query_vector = []
    amount_of_records = get_amount_of_records(inverted_index)
    for word in relevant_words:
        relevant_docs_amount = len(inverted_index[word])
        score = np.log(amount_of_records/relevant_docs_amount)
        query_vector.append(score)
    return query_vector


def calculate_similarity(query_vector,record_vector):
    return np.dot(query_vector,record_vector)/(np.linalg.norm(query_vector)*np.linalg.norm(record_vector))

query("vsm_inverted_index.json", "How are salivary glycoproteins from CF patients different from those of normal subjects?")
