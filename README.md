Multi-Stage Retrieval Pipeline for Question Answering


Project Overview:

This project implements a multi-stage text retrieval pipeline designed to efficiently handle question-answering (QA) tasks. By combining embedding models for candidate retrieval with cross-encoder models for reranking, this pipeline improves the accuracy of information retrieval, allowing for more relevant results in response to user queries.


Technologies and Concepts Used:

Datasets: Utilizes publicly available datasets from the BEIR benchmark, 
including:
Natural Questions (NQ)
HotpotQA
FiQA-2018

Embedding Model: nvidia/NV-Embed-v2 for initial candidate retrieval.
Cross-Encoder Model: cross-encoder/ms-marco-MiniLM-L-12-v2 for reranking documents based on relevance.
Metrics: Evaluated using NDCG@10 (Normalized Discounted Cumulative Gain).


Installation Instructions:

1. Clone the Repository

To get started, clone this repository to your local machine:

git clone https://github.com/YourUsername/multi-stage-retrieval-pipeline.git
cd multi-stage-retrieval-pipeline

2. Set Up Virtual Environment

It's recommended to use a virtual environment to manage dependencies:

python -m venv .venv
# Activate the environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

3. Install Dependencies

Run the following command to install the required libraries:

pip install -r requirements.txt

4. Download the Datasets

Manually download the datasets from the BEIR benchmark page:

Visit this link: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/

From above, download the following datasets:
    1. Natural Questions (nq.zip)
    2. HotpotQA (hotpotqa.zip)
    3. FiQA (fiqa.zip)
Extract them into a directory structure like this:

        BotGauge/
        ├── NaturalQuestions/
        ├── HotpotQA/
        └── FiQA/


Running the Pipeline:

To execute the retrieval pipeline and see the results, run the following command:

python main.py


Pipeline Stages:

1. Dataset Loading: Loads the dataset (e.g., Natural Questions).
2. Candidate Retrieval: Retrieves the top-k relevant documents using the embedding model.
3. Reranking: Reranks the top-k documents using the cross-encoder model.
4. Evaluation: Calculates the NDCG@10 score to evaluate the ranking quality.


Evaluation Metrics:

The final output of the pipeline includes the NDCG@10 score, which measures the ranking quality based on relevance. Higher NDCG@10 scores indicate better retrieval performance.

Usage Example:
Here’s an example of how to run the pipeline for Natural Questions:

[Example_Code(Python)]:

from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import ndcg_score

# Step 1: Load the Dataset
data_path = "path_to_datasets/NaturalQuestions"
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# Step 2: Candidate Retrieval (Embedding Model)
embedding_model = SentenceTransformer('nvidia/NV-Embed-v2')
# Retrieve top-k documents and calculate cosine similarity...

# Step 3: Reranking (Cross-Encoder)
rank_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
# Rerank top-k documents...

# Step 4: Evaluation (NDCG@10)
# Calculate NDCG@10 for the retrieved and reranked results...


File Structure:

multi-stage-retrieval-pipeline/
├── datasets/                  # Store datasets here (downloaded manually)
├── main.py                    # Main script to run the pipeline
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation


Contributing:

If you would like to contribute to this project, feel free to fork the repository and submit pull requests. Suggestions and improvements are welcome!


License:

This project is licensed under the MIT License.