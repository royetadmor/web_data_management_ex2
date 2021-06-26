import os
from lxml import etree


def build_inverted_index(path):
    xml_files = get_xml_files_data(path)
    data = extract_data_from_files(xml_files)


"""
Given the loaded xml files, returns a list of dictionary, containing:
    1. paper_num - Unique ID of each paper
    2. title - title of each document
    3. text - A textual summary of the document. Each document either has a "abstract" or "extract" part,
              Containing the textual summary.
"""


def extract_data_from_files(files):
    documents_identifier = []
    i = 0
    for file in files:
        for paper_num in file.xpath("//root//RECORD//PAPERNUM//text()"):
            documents_identifier.append({"paper_num":paper_num})
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


build_inverted_index("document_corpus")
