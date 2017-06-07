# BMASS

Source code used for ACL BioNLP 2017 workshop paper:

Denis Newman-Griffis, Albert M Lai, Eric Fosler-Lussier. [_Insights into Analogy Completion from the Biomedical Domain_](http://web.cse.ohio-state.edu/~newman-griffis.1/papers/2017-BioNLP.pdf)

## Dataset

[Download the dataset](http://slate.cse.ohio-state.edu/UTSAuthenticatedDownloader/index.html?dataset=BMASS) (requires a valid [UMLS Terminology Services](https://uts.nlm.nih.gov//license.html) login).

## Code

+ `analogy_task`: implementation of the analogy task (using TensorFlow v0.7)
+ `BMASS`: parser for BMASS data files
+ `lib`: various dependencies

A demo virtual machine setup is also included in the `demo` directory, using [Vagrant](https://www.vagrantup.com/).  This will run the analogy experiment for the full
BMASS dataset on CBOW and skip-gram embeddings pre-trained on the 2016 PubMed baseline.

To run the demo:
```bash
cd demo
vagrant up
vagrant ssh
$ cd /vagrant/src
$ make demo
```

## PubMed embeddings

The embeddings trained for this paper can be downloaded from [here](http://slate.cse.ohio-state.edu/BMASS).
