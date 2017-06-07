DATA = '/vagrant/data'
FREQUENT_TERMS = '/vagrant/data/PubMed_frequent_terms.txt'
LABELED_EMBEDDINGS = [
    (
        '/vagrant/data/embeddings/PubMed_CBOW.bin',
        'PubMed CBOW',
        False
    ),
    (
        '/vagrant/data/embeddings/PubMed_SGNS.bin',
        'PubMed SGNS',
        False
    )
]
    
if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'DATA':
        print(DATA)
