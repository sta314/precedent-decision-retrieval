from flask import Flask, render_template, Response
import numpy as np
import pandas as pd
from Inference import Inference

app = Flask(__name__)

legal_documents = pd.read_pickle('processed_dataset/dataset_full_html.pkl')

last_query = '---'
doc_scores = np.zeros((1, 1))
doc_indices = np.zeros((1, 1))

print("Initializing Inferer")

inferer = Inference()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search/<query>/<page>')

def search(query, page):

    global doc_scores, doc_indices, last_query

    if last_query != query:
        doc_scores, doc_indices = inferer.score(query, False)
        last_query = query
    
    start = (int(page) - 1) * 10
    end = start + 10
    matches, scores = get_n_matches(start, end)
    return {'matches': matches, 'scores': scores}

def get_n_matches(start, end):
    global legal_documents, doc_scores

    return legal_documents.iloc[doc_indices[start:end]].extracted_data.str.replace('\\t', '    ').tolist(), [str(score) for score, _ in doc_scores[start:end]]

@app.route('/document/<doc_id>')

def retrieve_document(doc_id):  
    global legal_documents

    doc = legal_documents.iloc[doc_indices[int(doc_id) - 1]].extracted_data.replace('\\t', '    ')
    return Response(doc, mimetype='text/html')

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
