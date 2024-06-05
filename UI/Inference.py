
import pandas as pd
import gensim as gs
import numpy as np
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import Stemmer
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, OkapiBM25Model
from gensim.similarities import SparseMatrixSimilarity
from sklearn.preprocessing import MinMaxScaler

class Inference:
    def __init__(self):
        
        # Read dataset, original to be used in BERT - processed to be used in BM25

        self.original_df = pd.read_pickle("processed_dataset/dataset_qdpairs_raw.pkl")
        self.processed_df = pd.read_pickle("processed_dataset/dataset_qdpairs_processed.pkl")


        # Load up BM25 pickles

        self.corpus = pickle.load(open('gensim_bm25_pickles/corpus.pkl', 'rb'))
        self.dictionary = pickle.load(open('gensim_bm25_pickles/dictionary.pkl', 'rb'))
        self.bm25_model = pickle.load(open('gensim_bm25_pickles/bm25_model.pkl', 'rb'))
        self.bm25_corpus = pickle.load(open('gensim_bm25_pickles/bm25_corpus.pkl', 'rb'))
        self.bm25_index = pickle.load(open('gensim_bm25_pickles/bm25_index.pkl', 'rb'))


        # Load up BERTurk model and pre-computed document embeddings

        self.model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')
        self.model.max_seq_length = 512
        self.model.to('cuda')

        self.doc_embeddings = np.load('doc_embeddings_npy/doc_embeddings.npy')


        self.bm25_index.index


    # Define the function for preprocessing query for BM25

    def preprocess_str(self, str_to_process):
        # This is a turkish stemmer, doesn't work perfect but it is consistent at least
        stemmer = Stemmer.Stemmer('turkish')
        
        str_result = str_to_process
        # Remove non-chars
        str_result = gs.parsing.preprocessing.strip_multiple_whitespaces(gs.parsing.preprocessing.strip_numeric(gs.parsing.preprocessing.strip_non_alphanum(str_result)))
        # Lowercase str
        str_result = str_result.lower()
        # Remove stopwords
        str_result = gs.parsing.preprocessing.remove_stopwords(s=str_result, stopwords=stopwords.words("turkish"))
        # Split str
        str_result = str_result.split()
        # Stem words
        str_result = stemmer.stemWords(str_result)

        return str_result

    # Define functions for scoring each document given query

    def score(self, query, is_index = False):

        query_bm25 = query if not is_index else self.processed_df.loc[query].query
        query_bert = query if not is_index else self.original_df.loc[query].query

        bm25_scores = self.score_BM25(query_bm25, not is_index)
        bert_scores = self.score_BERT(query_bert)

        bm25_scores = [score for score, _ in bm25_scores]
        bert_scores = [score for score, _ in bert_scores]

        # Normalize the scores using MinMaxScaler
        scaler = MinMaxScaler()
        bm25_scores_normalized = scaler.fit_transform(np.array(bm25_scores).reshape(-1, 1))
        bert_scores_normalized = scaler.fit_transform(np.array(bert_scores).reshape(-1, 1))

        # Average the normalized scores
        average_scores = 0.5 * bm25_scores_normalized + 0.5 * bert_scores_normalized

        # Combine the average scores with the document indices
        sorted_indices = np.argsort(average_scores.flatten())[::-1]
        final_scores = list(zip(average_scores.flatten()[sorted_indices], self.processed_df.index[sorted_indices]))

        return final_scores, sorted_indices
        

    def score_BM25(self, query, preprocess = True):
        preprocessed_query = self.preprocess_str(query) if preprocess else query
        tfidf_model = TfidfModel(dictionary=self.dictionary, smartirs='bnn')  # Enforce binary weighting of queries
        tfidf_query = tfidf_model[self.dictionary.doc2bow(preprocessed_query)]
        
        similarities = self.bm25_index[tfidf_query]

        scores = [(similarity, index) for similarity, index in zip(similarities, self.processed_df.index)]
        # scores = sorted(zip(similarities, processed_df.index), reverse=True)

        return scores

    def score_BERT(self, query):
        query_embedding = self.model.encode(query, normalize_embeddings=True)

        similarities = np.dot(self.doc_embeddings, query_embedding.T)

        scores = [(similarity, index) for similarity, index in zip(similarities, self.original_df.index)]
        # scores = sorted(zip(similarities, original_df.index), reverse=True)

        return scores