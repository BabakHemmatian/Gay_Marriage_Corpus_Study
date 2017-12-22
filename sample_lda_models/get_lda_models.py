from gensim import corpora
from gensim.models.ldamodel import LdaModel
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    level=logging.INFO, filename='ldamodel_log')

for i in (1,2):
    f='sampletext{}.txt'.format(i)
    lemmas=[]
    with open(f, 'r') as fh:
        line=fh.readline().strip()
        while line.strip():
            lemmas.append(line.split())
            line=fh.readline().strip()
    id2word=corpora.Dictionary(lemmas)
    corpus=[ id2word.doc2bow(lemmas_) for lemmas_ in lemmas ]
    ldamodel=LdaModel(corpus, id2word=id2word, num_topics=2)
    ldamodel.save('ldamodel{}'.format(i))
