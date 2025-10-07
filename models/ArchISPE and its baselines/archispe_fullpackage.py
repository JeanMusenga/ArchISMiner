#!pip install contractions colorama rouge-score swifter

"""# Import Important Libraries"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
import string
from nltk.tokenize import sent_tokenize
import re
from rouge_score import rouge_scorer
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import torch.optim as optim
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from ipywidgets import widgets
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Downloading necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

"""## Download and Load BERTOverflow Model"""

# Initialize tokenizer and model from 'jeniya/BERTOverflow'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("jeniya/BERTOverflow")
model = AutoModel.from_pretrained("jeniya/BERTOverflow").to(device)

"""# I. Post Preprocessing Layer"""

# Applying heuristic techinques to reduce noice in the data
def clean_dataset(text):
    if not isinstance(text, str):
        return text

    soup = BeautifulSoup(text, "html.parser")

    # Replace all links with '[external-link]'
    for a in soup.find_all('a'):
        a.replace_with('[external-link]')

    # Replace all images with '[figure]'
    for img in soup.find_all('img'):
        img.replace_with('[figure]')

    # Replace all code blocks with '[code-snippet]'
    for code in soup.find_all('code'):
        code.replace_with('[code-snippet]')

    # Replace all tables with '[table]'
    for table in soup.find_all('table'):
        table.replace_with('[table]')

    # Get the text without any remaining HTML tags
    clean_text = soup.get_text()

    return clean_text

dataset = pd.read_excel('367_ARPs_for_extracting_Issue_Solution_Pairs.xlsx')
# Apply the function to 'Question_body' and 'Answer_body' columns
dataset['Question_body_cleaned'] = dataset['Question_body_cleaned'].apply(clean_dataset)
dataset['Answer_body_cleaned'] = dataset['Answer_body_cleaned'].apply(clean_dataset)

dataset[['Question_body_cleaned', 'Answer_body_cleaned']].head()

# Tokenization, Lemmatization, and Stopword Removal
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    sentences = sent_tokenize(text)
    stop_words = set(ENGLISH_STOP_WORDS)
    processed_sentences = []
    for sentence in sentences:

        words = re.findall(r'\b\w+\b', sentence.lower())
        words = [word for word in words if word not in stop_words and word.isalpha()]

        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        processed_sentences.append(" ".join(lemmatized_words))
    return sentences, processed_sentences

# Preprocess question and answer bodies
dataset['processed_question'] = dataset['Question_body_cleaned'].apply(preprocess_text)
dataset['processed_answer'] = dataset['Answer_body_cleaned'].apply(preprocess_text)

"""# II. Feature Extraction Layer

---

## 1. Contextual Feature Extractor
"""

# Define BERTOverflow embedding extraction
def get_bert_embeddings(sentences):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract embeddings from the [CLS] token for each sentence and get mean of token embeddings for each sentence
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

# Get BERT embeddings from preprocessed question and answer text
dataset['question_embeddings'] = dataset['processed_question'].apply(lambda x: get_bert_embeddings(x[1]))
dataset['answer_embeddings'] = dataset['processed_answer'].apply(lambda x: get_bert_embeddings(x[1]))

# Check shape of embeddings in the first 3 rows
for i in range(3):
    emb = dataset['question_embeddings'].iloc[i]
    print(f"Row {i} - question embedding shape: {emb.shape}")

# Ensure all sentence embeddings are the same size (should be 768 for BERT)
vector_sizes = dataset['question_embeddings'].apply(lambda x: [vec.shape[0] for vec in x])
print(vector_sizes.tolist())

"""## 2. Local Feature Extractor

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define TextCNN model with adjusted kernel sizes
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, filter_sizes=[2, 2, 2], num_filters=100, num_classes=256):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (filter_size, embedding_dim)) for filter_size in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        # Shape: (batch_size, seq_length, embedding_dim)
        x = self.embedding(x)
        x = x.unsqueeze(1)

        # Apply convolution and squeeze the last dimension, resulting in a shape of (batch_size, num_filters, seq_length)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # pply max pooling across the sequence length dimension, resulting in a shape of (batch_size, num_filters)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        # Concatenate pooled outputs (batch_size, num_filters * len(filter_sizes))
        x = torch.cat(x, 1)
        # Fully connected layer
        x = self.fc(x)
        return x

# Define a function to extract TextCNN features
def extract_textcnn_features(sentences, model, tokenizer, device):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        # Extract learned TextCNN features from tokenized input
        features = model(inputs['input_ids'])
    return features.cpu().numpy()

# Initialize TextCNN model with the adjusted kernel sizes
vocab_size = len(tokenizer.vocab)

# embedding dimension
embedding_dim = 256
textcnn_model = TextCNN(vocab_size, embedding_dim, filter_sizes=[2, 2, 2], num_classes=256).to(device)

"""## 3. Linguistic Pattern Extractor"""

# Define fucntion for question_thread Domain-Specific, Heuristic, Linguistic Feature Extraction
def q_extract_domain_specific_linguistic_patterns_heuristic_features(sentence, architectural_keywords, fiveW1H_keywords_question, linguistic_patterns):
    features = {'contains_architecture_keywords': 0, 'contains_question_words': 0, 'contains_linguistic_patterns': 0}
    for keyword in architectural_keywords:
        if keyword.lower() in sentence.lower():
            features['contains_architecture_keywords'] = 1
            break
    for keyword in fiveW1H_keywords_question:
        if keyword.lower() in sentence.lower():
            features['contains_question_words'] = 1
            break
    for pattern in linguistic_patterns:
        if pattern.lower() in sentence.lower():
            features['contains_linguistic_patterns'] = 1
            break
    return features


# Define fucntion for answer_thread Domain-Specific, Heuristic, Linguistic Feature Extraction
def a_extract_domain_specific_linguistic_patterns_heuristic_features(sentence, architectural_keywords, linguistic_patterns):
    features = {'contains_architecture_keywords': 0, 'contains_linguistic_patterns': 0}
    for keyword in architectural_keywords:
        if keyword.lower() in sentence.lower():
            features['contains_architecture_keywords'] = 1
            break
    for pattern in linguistic_patterns:
        if pattern.lower() in sentence.lower():
            features['contains_linguistic_patterns'] = 1
            break
    return features

# List of architectural keywords categorized for clarity
architectural_keywords = {
    "Architectural Patterns and Styles": [
        "Architecture pattern", "Design pattern", "MVC", "Model View Controller", "Monolith",
        "Microservice", "microservices", "MVP", "Model View Presenter", "MVVP",
        "Model View ViewModel", "MVVM", "Client-Server", "Client Server", "Client/Server",
        "Layered pattern", "N-Tier", "Event Driven pattern", "Event Driven",
        "Pipe and Filter", "Service Oriented Architecture", "SOA", "Broker", "Peer to Peer",
        "Master-Slave", "Master and Slave", "Blackboard", "Command Query Responsibility Segregation", "CQRS"
        "Hexagonal Architecture", "Hexagonal", "Publish–Subscribe", "Publish and Subscribe", "Event Sourcing",
        "Reactive Architecture", "Database Per Service", "Pipe-and-Filter with Feedback Loops",
        "Saga Pattern", "Service Mesh Architecture", "Strangler Fig Pattern",
        "Multi-Tenant Architecture", "Actor Model", "Interpreter Architecture",
        "Pipeline Architecture", "Digital Twin Architecture", "User Interface"
        "Monolithic", "Event-Driven", "Hybrid Architecture", "Clean Architecture"
    ],

    "Architectural Tactics": [
        "Architecture tactic", "Design tactic", "Heartbeat", "Checkpoint", "Checkpointing",
        "Retry Mechanism", "Failover Mechanism", "Load Balancing", "Caching", "Concurrency",
        "Queue-Based Load Management", "Data Compression", "Lazy Loading", "Authentication",
        "Authorization", "Data Encryption", "Intrusion Detection", "Audit Logging",
        "Firewalls", "API Gateways", "Cache", "Caching","Loose coupling","Resource Pooling",
        "Failover"
    ],

    "Software Design Principles": [
        "Encapsulation", "Separation of Concerns", "Abstraction",
        "Component-Based Design", "Refactoring", "Plug-in Architecture"
    ],

    "Scalability and Performance Optimization": [
        "Horizontal Scaling", "Scale-Out", "Vertical Scaling", "Scale-Up",
        "Sharding", "Database Replication", "Progressive Disclosure","Server Replication",
        "Undo Mechanism", "Redo Mechanism", "Event-Bus Pattern"
    ],

    "Reliability and Fault Tolerance": [
        "Consistent UI Design", "Removal from service", "Exception Prevention",
        "Introduce Concurrency", "Maintain Multiple Copies of Data", "Bound Queue Sizes",
        "Schedule Resources", "Manage Resources", "Manage Sampling Rate", "Limit Event Response",
        "Prioritize Events", "Bound Execution Times", "Increase Resource Efficiency"
    ],

    "Networking and Communication": [
        "REST", "SOAP", "WCF", "Ping/Echo", "Ping and Echo","Shadow", "Active Redundancy", "Monitor",
        "Timestamp", "Sanity Checking", "Voting", "Condition Monitoring"
    ],

    "Error Handling and Recovery": [
        "Degradation", "Retry", "Ignore Faulty Behavior", "Rollback",
        "Exception Handler", "Spare", "Non-Stop Forwarding", "State Resynchronization"
    ],

    "Security Strategies": [
        "Increase Resources", "Maintain Multiple Copies of Computations",
        "Detect Intrusion", "Detect Service Denial", "Verify Message Integrity",
        "Detect Message Delay", "Identify Actors", "Limit Access",
        "Limit Exposure", "Encrypt Data"
    ],


    "Modularity and Maintainability": [
        "Tailor Interface", "Reduce Size of a Module", "Split Module",
        "Increase Cohesion", "Increase Semantic Coherence", "Reduce Coupling",
        "Encapsulate", "Use an Intermediary", "Restrict Dependency", "Refactor",
        "Abstract Common Services", "Reduce Overhead", "Limit Nondeterminism",
        "Limit Structural Complexity", "Limit Complexity"
    ],

    "User Experience and Usability": [
        "Specialized Interface", "Record/Playback", "Localize State Storage",
        "Abstract Data Sources", "Sandbox", "Executable Assertions",
        "Support User Initiative", "Support System Initiative",
        "Maintain Task Model", "Maintain User Model", "Aggregate", "Maintain System"
    ],

    "Architecture Design Decision": [
        "Architecture decision", "Trade-offs", "Requirements", "MongoDB", "Redis"
        , "Redis" , "MySQL" , "PostgreSQL" , "SQL Server", "Amazon DynamoDB", "TimescaleDB", "InfluxDB"
    ],

    "Design Context": [
        "Embedded System", "Mobile Application", "Web Application",
        "Information System", "Game application", "E-commerce", "Distributed System",
        "Banking System", "Android", "iOS", "Window",
    ],

    "Maintainability": [
        "Maintainability", "Update", "Modify", "Modular", "Decentralized",
        "Encapsulation", "Dependency", "Readability", "Interdependent",
        "Understandability", "Modifiability", "Modularity", "Maintain",
        "Analyzability", "Changeability", "Testability"
    ],

    "Performance (Efficiency)": [
        "Performance", "Processing time", "Response time", "Resource Consumption",
        "Throughput", "Efficiency", "Operation", "Achievement", "Interaction",
        "Accomplishment", "Parallelism"
    ],

    "Compatibility": [
        "Compatibility", "Co-existence", "Interoperability", "Exchange", "Sharing"
    ],

    "Usability": [
        "Usability", "Flexibility", "Interface", "User-friendly",
        "Configurable", "Serviceability", "Accessibility", "Customizable"
    ]
}
# this can be extanded during the future research

#Heuristic features
fiveW1H_keywords_question = ["What", "When", "Who", "Which", "How", "?"]


#Linguistic Pattern features
linguistic_patterns = ["I am trying", "I want to design", "How to architecture", "I want to design", "I'm designing",
                       "I'm building","The user should", "I need help","I am developing","Advise on","I recommend","I cannot recommend",
                       "design an application","the best practice", "I'm trying to", "I'm having a hard", "help", "you should","I am using", "you don't have to do","In order to ","it is critical",
                       "You should","It is recommended","A good approach is"
                       ]

"""# III. Similarity and Relevancy Assessment Layer

## Function for identifying relevant and important issue sentences within a question body
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def question_score_sentences_with_combined_features(question_sentences, question_embeddings,
                                                    textcnn_features, weight_bert=0.4, weight_textcnn=0.2, weight_heuristic=0.1,
                                                    architectural_keywords=[], fiveW1H_keywords_question=[], linguistic_patterns=[]):

    q_sentence_scores = []

    # Optional: Represent full question as mean of all sentence embeddings
    full_question_embedding = np.mean(question_embeddings, axis=0)

    for i, (q_sentence, tcnn_features) in enumerate(zip(question_sentences, textcnn_features)):

        # BERT-based Score using cosine similarity
        if i < len(question_embeddings):
            bert_score = cosine_similarity(
                [question_embeddings[i]], [full_question_embedding])[0][0]
        else:
            bert_score = 0

        # TextCNN score
        textcnn_score = weight_textcnn * np.mean(tcnn_features) if isinstance(tcnn_features, np.ndarray) and tcnn_features.size > 0 else 0

        # Heuristic Score (e.g., sentence length)
        sentence_length = len(q_sentence.split())
        max_length = max(len(s.split()) for s in question_sentences)
        heuristic_score = sentence_length / max_length if max_length > 0 else 0


        # Extract domain-specific, heuristic, linguistic features
        q_features = q_extract_domain_specific_linguistic_patterns_heuristic_features(
            q_sentence, architectural_keywords, fiveW1H_keywords_question, linguistic_patterns)


        # Combine all features
        q_combined_score = (weight_heuristic * heuristic_score) + \
                           (weight_bert * bert_score) + \
                           (weight_textcnn * textcnn_score) + \
                           (q_features['contains_architecture_keywords'] * 0.1) + \
                           (q_features['contains_question_words'] * 0.1) + \
                           (q_features['contains_linguistic_patterns'] * 0.1)

        q_sentence_scores.append(q_combined_score)

    return q_sentence_scores

dataset['question_scores'] = dataset.apply(lambda x: question_score_sentences_with_combined_features(
    x['processed_question'][0],
    x['question_embeddings'],    # list of embeddings for question sentences
    extract_textcnn_features(x['processed_question'][0], textcnn_model, tokenizer, device),
    architectural_keywords=architectural_keywords,
    fiveW1H_keywords_question=fiveW1H_keywords_question,
    linguistic_patterns=linguistic_patterns
), axis=1)

"""## Function for identifying relevant and important solution sentences within a answer body"""

def answer_score_sentences_with_combined_features(question_sentences, answer_sentences, question_embeddings, answer_embeddings,
                                                  textcnn_features, weight_bert=0.5, weight_textcnn=0.2, weight_heuristic=0.1,
                                                  architectural_keywords=[], fiveW1H_keywords_question=[], linguistic_patterns=[]):
    a_sentence_scores = []

    for a_sentence, a_embedding, tcnn_features in zip(answer_sentences, answer_embeddings, textcnn_features):
        a_embedding = a_embedding.reshape(1, -1)  # Reshape for cosine similarity
        similarities = [cosine_similarity(a_embedding, q_embedding.reshape(1, -1))[0][0] for q_embedding in question_embeddings]
        q_a_bert_score = np.mean(similarities)

        # Ensure TextCNN features are valid before computing score
        if isinstance(tcnn_features, np.ndarray) and tcnn_features.size > 0:
            textcnn_score = weight_textcnn * np.mean(tcnn_features)  # Use mean pooling instead of tcnn_features[0]
        else:
            textcnn_score = 0  # Default to zero if features are missing or empty

        # Heuristic Score (e.g., sentence length)
        a_sentence_length = len(a_sentence.split())
        a_heuristic_score = a_sentence_length / max(len(s.split()) for s in question_sentences)

        # Extract domain-specific, heuristic, Linguistic feature Extraction
        a_donain_linguistic_patterns_heuristic_features = q_extract_domain_specific_linguistic_patterns_heuristic_features(
            a_sentence, architectural_keywords, fiveW1H_keywords_question, linguistic_patterns)

        # Combine features with respective weights
        a_combined_score = (weight_bert * q_a_bert_score) + (weight_heuristic * a_heuristic_score) + \
                           (weight_textcnn * textcnn_score) + \
                           (a_donain_linguistic_patterns_heuristic_features['contains_architecture_keywords'] * 0.1) + \
                           (a_donain_linguistic_patterns_heuristic_features['contains_linguistic_patterns'] * 0.1)

        a_sentence_scores.append(a_combined_score)

    return a_sentence_scores

dataset['answer_scores'] = dataset.apply(lambda x: answer_score_sentences_with_combined_features(
    x['processed_question'][0],
    x['processed_answer'][0],
    x['question_embeddings'],
    x['answer_embeddings'],

    extract_textcnn_features(x['processed_answer'][0], textcnn_model, tokenizer, device),
    architectural_keywords=architectural_keywords,
    linguistic_patterns=linguistic_patterns),
axis=1)

"""# V. Output Layer: Sentence Importance Ranking and Extraction"""

import numpy as np
#  Extracts the top-k most important sentences based on model scores mwhile preserving their original order in the text.
def issue_solution_extraction(original_sentences, scores, num_sentences=6):

    # Handle cases where there are fewer sentences than requested
    num_sentences = min(num_sentences, len(original_sentences))

    # Step 1: Get indices of top-k sentences by descending score
    ranked_sentence_indices = np.argsort(scores)[::-1][:num_sentences]

    # Step 2: Sort those indices by their original order
    ordered_indices = sorted(ranked_sentence_indices)

    # Step 3: Extract and concatenate sentences in natural order
    issue_solution = [original_sentences[idx] for idx in ordered_indices]
    return " ".join(issue_solution)

# Apply the extraction function to both question and answer parts
dataset['Issue_Extracted'] = dataset.apply(
    lambda x: issue_solution_extraction(
        x['processed_question'][0],
        x['question_scores']
    ),
    axis=1
)

dataset['Solution_Extracted'] = dataset.apply(
    lambda x: issue_solution_extraction(
        x['processed_answer'][0],
        x['answer_scores']
    ),
    axis=1
)

# Save the dataset with the extracted issues and solutions
try:
    dataset.to_excel('/content/ArchISPE_run_Results.xlsx', index=False, engine='openpyxl')
    print("File saved successfully!")
except Exception as e:
    print(f"An error occurred: {e}")

# Display the extracted issues and solutions
issue_solutions = dataset[['Question_title', 'Issue_Extracted', 'Solution_Extracted']]
issue_solutions.head()

"""# Evaluate the extracted issue-solution pairs using Precsion, Recall, and F1"""

from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_extracted_Issue_solutions_at_sentence_level(df, ref_col, gen_col):
    precision_list = []
    recall_list = []
    f1_list = []

    for index, row in df.iterrows():
        ref_issue_solution = row[ref_col]
        gen_issue_solution = row[gen_col]

        if pd.isna(ref_issue_solution) or pd.isna(gen_issue_solution):
            continue

        ref_sentences = nltk.sent_tokenize(ref_issue_solution)
        gen_sentences = nltk.sent_tokenize(gen_issue_solution)

        ref_sentences_set = set(ref_sentences)
        gen_sentences_set = set(gen_sentences)

        precision = len(ref_sentences_set & gen_sentences_set) / len(gen_sentences_set) if gen_sentences_set else 0
        recall = len(ref_sentences_set & gen_sentences_set) / len(ref_sentences_set) if ref_sentences_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    metrics_df = pd.DataFrame({
        'precision': precision_list,
        'recall': recall_list,
        'f1': f1_list
    })

    return metrics_df

# Load your DataFrame
#df = pd.read_excel('ArchISPE_BERTOverflow_TextCNN_Domain_Heuristics_Features.xlsx')
df = pd.read_excel('ArchISPE_results.xlsx')

question_metrics_df = evaluate_extracted_Issue_solutions_at_sentence_level(df, 'Ground_truth_Issue_Labeled', 'Issue_Extracted')
answer_metrics_df = evaluate_extracted_Issue_solutions_at_sentence_level(df, 'Ground_truth_Solution_Labeled', 'Solution_Extracted')

# Separate evaluation for question and answer summaries, retaining individual columns for comparison
question_metrics_df.columns = [f'Question_{col}' for col in question_metrics_df.columns]
answer_metrics_df.columns = [f'Answer_{col}' for col in answer_metrics_df.columns]

# Combine question and answer results into a single DataFrame
combined_metrics_df = pd.concat([question_metrics_df, answer_metrics_df], axis=1)

# Compute overall Precision, Recall, F1 scores (separately for questions and answers)
mean_question_metrics = question_metrics_df.mean()
mean_answer_metrics = answer_metrics_df.mean()

print("\nMean Precision, Recall, F1 Scores for \033[31mQuestions\033[0m:")
print(mean_question_metrics)

print("\nMean Precision, Recall, F1 Scores for \033[31mAnswers\033[0m:")
print(mean_answer_metrics)

Mean Precision, Recall, F1 Scores for Questions:
Question_precision    0.883893
Question_recall       0.884953
Question_f1           0.883493
dtype: float64

Mean Precision, Recall, F1 Scores for Answers:
Answer_precision    0.897541
Answer_recall       0.891663
Answer_f1           0.893854
dtype: float64