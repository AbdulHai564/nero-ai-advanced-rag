from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from retrieval import semantic_retrieval, bm25_retrieval, hybrid_retrieval, cohere_reranker, fetch_parents, generate_answer
from ingest import load_chunk

def run_eval(path,test_cases):
    parent_chunks,child_chunks=load_chunk(path)
    retriever=semantic_retrieval()
    bm25_retriever=bm25_retrieval(child_chunks)

    def process(case):
        q=case["question"]
        hybrid_results=hybrid_retrieval(bm25_retriever,retriever,q)
        reranked=cohere_reranker(hybrid_results,q)
        parents=fetch_parents(reranked)
        answer=generate_answer(parents,q)
return {
            "question": q,
            "answer": answer,
            "contexts": [doc.page_content for doc in parents],
            "ground_truth": case["ground_truth"]
        }


