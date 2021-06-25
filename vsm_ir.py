import sys
from os import path
import inverted_index_builder
import search_engine



"""
Implementation of command line interface in the requested format.
Can accept one of the two command formats:
    1. python3 vsm_ir.py create_index <path_to_xml>
    2. python3 vsm_ir.py query <path_to_inverted_index> <question>
    
Any other format will print an error message and exit.
"""


def main():
    args = sys.argv[1:]
    if(len(args) == 2):
        command = args[0]
        path_to_file = args[1]
        if command == "create_index" and path.exists(path_to_file):
            inverted_index_builder.build_inverted_index(path_to_file)
        else:
            print("No such command or path to file is wrong.\nExiting.")
            exit(1)
    elif(len(args) == 3):
        command = args[0]
        path_to_file = args[1]
        question = args[2]
        if(command == "query" and path.exists(path_to_file)):
            search_engine.query(path_to_file,question)
        else:
            print("No such command or path to file is wrong.\nExiting.")
            exit(1)
    else:
        print("No such command.\nExiting")
        exit(1)











if __name__ == "__main__":
    main()