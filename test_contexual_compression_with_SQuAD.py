import os
import json
import torch
import concurrent.futures
from uuid import uuid4
from tqdm import tqdm
import faiss
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS as FAISSVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
import numpy as np
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


os.environ["GOOGLE_API_KEY"] = "AIzaSyCMhrggDcsDo7d7dRmDmppelr1D96kiXN4"

# Load dataset from JSON file
def load_json_dataset(filename="dev-v2.0.json"):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            return dataset
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        raise
    except json.JSONDecodeError:
        print(f"Error: File {filename} contains invalid JSON.")
        raise

# Create text file from dataset, handling SQuAD structure
def create_text_file(dataset, filename="SQuAD_v2.txt"):
    with open(filename, 'w', encoding='utf-8') as f:
        if 'data' in dataset:
            for article in dataset['data']:
                title = article.get('title', 'Untitled')
                f.write(f"# {title}\n\n")
                for para in article.get('paragraphs', []):
                    f.write(f"## Context\n{para['context']}\n\n")
                    f.write("## Questions\n")
                    for qa in para.get('qas', []):
                        answers = qa.get('answers', qa.get('plausible_answers', []))
                        answer_texts = [ans['text'] for ans in answers] if answers else ["No answer (impossible question)"]
                        f.write(f"- Q: {qa['question']}\n  A: {', '.join(answer_texts)}\n")
                    f.write("\n")
        else:
            title = dataset.get('title', 'Untitled')
            f.write(f"# {title}\n\n")
            for para in dataset.get('paragraphs', []):
                f.write(f"## Context\n{para['context']}\n\n")
                f.write("## Questions\n")
                for qa in para.get('qas', []):
                    answers = qa.get('answers', qa.get('plausible_answers', []))
                    answer_texts = [ans['text'] for ans in answers] if answers else ["No answer (impossible question)"]
                    f.write(f"- Q: {qa['question']}\n  A: {', '.join(answer_texts)}\n")
                f.write("\n")

# Function to print documents
def pretty_print_docs(docs, is_compressed=False):
    docs_to_print = docs[:4] if is_compressed else docs
    return "\n{'-' * 100}\n".join([f"Document {i+1}:\n\n{d.page_content}" for i, d in enumerate(docs_to_print)])

# Function to calculate Compression Efficiency Score (CES)
def calculate_ces(original_docs, compressed_docs, similarity_threshold, max_similarity=1.0, w1=0.25, w2=0.25, w3=0.5):
    orig_doc_count = len(original_docs)
    comp_doc_count = len(compressed_docs)
    dcr = max(0, (orig_doc_count - comp_doc_count) / orig_doc_count) if orig_doc_count > 0 else 0
    orig_length = sum(len(doc.page_content) for doc in original_docs)
    comp_length = sum(len(doc.page_content) for doc in compressed_docs)
    clr = (orig_length - comp_length) / orig_length if orig_length > 0 else 0
    rs = similarity_threshold / max_similarity
    ces = w1 * dcr + w2 * clr + w3 * rs
    return ces, dcr, clr, rs

# Function to evaluate retrieval performance
def evaluate_retrieval(docs, question, answers, context):
    relevant = False
    for ans in answers:
        if ans['text'] in context and any(ans['text'] in doc.page_content for doc in docs):
            relevant = True
            break
    retrieved = len(docs)
    relevant_retrieved = 1 if relevant else 0
    relevant_total = 1  # Each question has one relevant context
    precision = relevant_retrieved / retrieved if retrieved > 0 else 0
    recall = relevant_retrieved / relevant_total if relevant_total > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

# Function to calculate information loss
def calculate_information_loss(original_docs, compressed_docs, qas, context):
    orig_length = sum(len(doc.page_content) for doc in original_docs)
    comp_length = sum(len(doc.page_content) for doc in compressed_docs)
    content_coverage = comp_length / orig_length if orig_length > 0 else 0
    total_answers = sum(len(qa['answers']) for qa in qas if not qa['is_impossible'])
    retained_answers = 0
    for qa in qas:
        if qa['is_impossible']:
            continue
        for ans in qa['answers']:
            if any(ans['text'] in doc.page_content for doc in compressed_docs):
                retained_answers += 1
    answer_retention = retained_answers / total_answers if total_answers > 0 else 0
    return content_coverage, answer_retention

# Function to process a single question (for threading)
def process_question(qa, context, qas, retriever, compression_retriever):
    if qa['is_impossible']:
        return None
    question = qa['question']
    answers = qa['answers']

    # Naive retrieval
    naive_docs = retriever.invoke(question)
    naive_p, naive_r, naive_f1 = evaluate_retrieval(naive_docs, question, answers, context)

    # Compressed retrieval
    compressed_docs = compression_retriever.invoke(question)
    comp_p, comp_r, comp_f1 = evaluate_retrieval(compressed_docs, question, answers, context)

    # Compression Efficiency
    ces, dcr, clr, rs = calculate_ces(naive_docs, compressed_docs, similarity_threshold=0.5)

    # Information Loss
    content_coverage, answer_retention = calculate_information_loss(naive_docs, compressed_docs, qas, context)

    return {
        'naive': {'question': question, 'precision': naive_p, 'recall': naive_r, 'f1': naive_f1},
        'compressed': {'question': question, 'precision': comp_p, 'recall': comp_r, 'f1': comp_f1},
        'ces': {'question': question, 'ces': ces, 'dcr': dcr, 'clr': clr, 'rs': rs},
        'info_loss': {'question': question, 'content_coverage': content_coverage, 'answer_retention': answer_retention}
    }

# Main execution
def main():
    # Check CUDA availability
    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")
    device = 'cuda' if use_cuda else 'cpu'

    # Load dataset
    print("Loading dataset...")
    dataset = load_json_dataset()

    # Create text file
    print("Creating text file...")
    create_text_file(dataset)

    # Load and split documents
    print("Loading and splitting documents...")
    documents = TextLoader("SQuAD_v2.txt").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # For testing, use a small batch
    small_texts = texts[:10]  # Process only the first 10 documents

    # Initialize embeddings
    print("Initializing embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Determine embedding dimension dynamically
    sample_text = "Sample text to embed"
    sample_embedding = embeddings.embed_query(sample_text)
    embedding_dim = len(sample_embedding)

    # Create FAISS retriever with GPU support if CUDA is available
    print("Creating FAISS retriever...")
    if use_cuda:
        # Configure FAISS for GPU
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatL2(embedding_dim)
        #gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        faiss_vector_store = FAISSVectorStore.from_documents(
            small_texts,
            embeddings
        )
        res = faiss.StandardGpuResources()
        faiss_vector_store.index = faiss.index_cpu_to_gpu(res, 0, faiss_vector_store.index)
    else:
        # Use CPU-based FAISS
        faiss_vector_store = FAISSVectorStore.from_documents(
            small_texts,
            embeddings
        )
    retriever = faiss_vector_store.as_retriever(search_kwargs={"k": 4})

    # Set up document compression pipeline
    print("Setting up compression pipeline...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.5)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever, search_kwargs={"k": 4}
    )

    # Collect tasks for multithreading
    tasks = []
    articles = dataset['data'] if 'data' in dataset else [{'paragraphs': dataset.get('paragraphs', [])}]
    for article in articles:
        for para in article.get('paragraphs', []):
            context = para['context']
            for qa in para.get('qas', []):
                tasks.append((qa, context, para['qas']))

    # Process questions in parallel with progress bar
    naive_metrics = []
    compressed_metrics = []
    ces_metrics = []
    info_loss_metrics = []

    max_workers = 1
    print(f"Processing {len(tasks)} questions with {max_workers} threads on {device}...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_question, qa, context, qas, retriever, compression_retriever): qa
                          for qa, context, qas in tasks}
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="Processing questions"):
            result = future.result()
            if result:
                naive_metrics.append(result['naive'])
                compressed_metrics.append(result['compressed'])
                ces_metrics.append(result['ces'])
                info_loss_metrics.append(result['info_loss'])

    # Aggregate metrics
    print("Aggregating metrics...")
    def aggregate_metrics(metrics):
        avg_precision = sum(m['precision'] for m in metrics) / len(metrics) if metrics else 0
        avg_recall = sum(m['recall'] for m in metrics) / len(metrics) if metrics else 0
        avg_f1 = sum(m['f1'] for m in metrics) / len(metrics) if metrics else 0
        return avg_precision, avg_recall, avg_f1

    naive_avg_p, naive_avg_r, naive_avg_f1 = aggregate_metrics(naive_metrics)
    comp_avg_p, comp_avg_r, comp_avg_f1 = aggregate_metrics(compressed_metrics)
    avg_ces = sum(m['ces'] for m in ces_metrics) / len(ces_metrics) if ces_metrics else 0
    avg_dcr = sum(m['dcr'] for m in ces_metrics) / len(ces_metrics) if ces_metrics else 0
    avg_clr = sum(m['clr'] for m in ces_metrics) / len(ces_metrics) if ces_metrics else 0
    avg_rs = sum(m['rs'] for m in ces_metrics) / len(ces_metrics) if ces_metrics else 0
    avg_content_coverage = sum(m['content_coverage'] for m in info_loss_metrics) / len(info_loss_metrics) if info_loss_metrics else 0
    avg_answer_retention = sum(m['answer_retention'] for m in info_loss_metrics) / len(info_loss_metrics) if info_loss_metrics else 0

    # Generate report
    print("Generating report...")
    sample_question = articles[0]['paragraphs'][0]['qas'][0]['question'] if articles and articles[0].get('paragraphs') else "No question available"
    report = f"""
# Normans Retrieval Evaluation Report

## Retrieval Performance
### Naive Retrieval
- Average Precision: {naive_avg_p:.2%}
- Average Recall: {naive_avg_r:.2%}
- Average F1-Score: {naive_avg_f1:.2f}

### Compressed Retrieval
- Average Precision: {comp_avg_p:.2%}
- Average Recall: {comp_avg_r:.2%}
- Average F1-Score: {comp_avg_f1:.2f}

## Compression Efficiency
- Average CES: {avg_ces:.2f}
- Average Document Count Reduction (DCR): {avg_dcr:.2%}
- Average Content Length Reduction (CLR): {avg_clr:.2%}
- Average Relevance Score (RS): {avg_rs:.2f}

## Information Loss
- Average Content Coverage: {avg_content_coverage:.2%}
- Average Answer Retention: {avg_answer_retention:.2%}

## Sample Retrieval Output (for question: "{sample_question}")
### Naive Retrieval
{pretty_print_docs(retriever.invoke(sample_question))}

### Compressed Retrieval
{pretty_print_docs(compression_retriever.invoke(sample_question), is_compressed=True)}
"""
    with open("retrieval_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    # Print summary
    print("Retrieval evaluation completed. See 'retrieval_report.md' for detailed results.")
    print(f"Naive Retrieval - Precision: {naive_avg_p:.2%}, Recall: {naive_avg_r:.2%}, F1: {naive_avg_f1:.2f}")
    print(f"Compressed Retrieval - Precision: {comp_avg_p:.2%}, Recall: {comp_avg_r:.2%}, F1: {comp_avg_f1:.2f}")

if __name__ == "__main__":
    main()
