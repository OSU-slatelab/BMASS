DATA = 'data/'
FREQUENT_TERMS = 'data/PubMed_frequent_terms.txt'
LABELED_EMBEDDINGS = [
    (
        'data/embeddings/chiu-bionlp-2016/bio_nlp_vec/PubMed-shuffle-win-2.bin',
        'Chiu et al (win=2)',
        True
    ),
    (
        'data/embeddings/chiu-bionlp-2016/bio_nlp_vec/PubMed-shuffle-win-2.bin',
        'Chiu et al (win=30)',
        True
    ),
    (
        (
            'data/embeddings/PubMed_Glove.bin'
            'data/embeddings/PubMed_Glove_vocab'
        ),
        'PubMed Glove',
        False
    ),
    (
        'data/embeddings/PubMed_CBOW.bin',
        'PubMed CBOW',
        False
    ),
    (
        'data/embeddings/PubMed_SGNS.bin',
        'PubMed SGNS',
        False
    )
]
    
if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'DATA':
        print(DATA)
