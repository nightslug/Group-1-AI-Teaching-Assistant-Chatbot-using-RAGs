from sklearn.metrics.pairwise import cosine_similarity
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.documents import Document

import numpy as np
import streamlit as st




template= """You are an AI assistant specifically designed to work with RAG (Retrieval Augmented Generation) systems. Your responses must be strictly based on the provided context. Follow these rules:

1. Only answer using information explicitly present in the given context
2. If the context is not relevant to the query, respond with "The provided context is not relevant to answer this query."
3. If the context is partially relevant but incomplete, respond with "The context is incomplete to fully answer this query. Based on the available context, I can only tell you that: [insert relevant information from context]"
4. Do not make assumptions or add information beyond what's in the context
5. If you quote or reference specific parts of the context, indicate this in your response
6. If the context is empty or missing, respond with "No context was provided to answer this query."

Remember: You are a context-dependent system. Never draw from general knowledge - only use the specific context provided in each query.
Question: {question}
Context: {context}
Answer:"""



def process_query(query, embeddings, paragraph_embeddings, query_embedding=None, n_results=1):
    if query_embedding is None:
        query_embedding = embeddings.embed_query(query)
    similarities = cosine_similarity([query_embedding], paragraph_embeddings)
    most_similar_indices = np.argsort(similarities[0])[::-1][:n_results]
    return most_similar_indices, query_embedding

def naive_rag_process(text, chunk_size=512, embeddings=None, query=None):
    # Get query embedding first
    query_embedding = embeddings.embed_query(query)
    
    # Process text
    paragraphs = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    with st.spinner("Calculating embeddings..."):
        paragraph_embeddings = embeddings.embed_documents(paragraphs)
    paragraph_embeddings = np.array(paragraph_embeddings)
    
    query_embedding = embeddings.embed_query(query)
    
    return paragraphs, paragraph_embeddings, query_embedding

def semantic_chunking_rag_process(text, embeddings, query, llm):
    semantic_text_splitter = SemanticChunker(GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    semantic_chunks = semantic_text_splitter.create_documents([text])

    semantic_vector_store = FAISS.from_documents(semantic_chunks, embeddings)
    semantic_retriever = semantic_vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": semantic_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query)

def reranker_rag_process(text, embeddings, query, llm):
    semantic_text_splitter = SemanticChunker(GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    semantic_chunks = semantic_text_splitter.create_documents([text])

    semantic_vector_store = FAISS.from_documents(semantic_chunks, embeddings)
    semantic_retriever = semantic_vector_store.as_retriever()
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=semantic_retriever
    )

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": compression_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query)

def hybrid_rag_process(text, embeddings, query, llm):
    semantic_text_splitter = SemanticChunker(GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    semantic_chunks = semantic_text_splitter.create_documents([text])

    semantic_vector_store = FAISS.from_documents(semantic_chunks, embeddings)
    semantic_retriever = semantic_vector_store.as_retriever()

    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=semantic_retriever
    )

    bm25_retriever = BM25Retriever.from_documents([Document(page_content=text)])
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, compression_retriever],
        weights=[0.5, 0.5],
    )
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": ensemble_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query)

def self_rag_process(text,  embeddings, query, llm):
    # For now, same as naive implementation
    max_refinement_rounds = 2
    semantic_text_splitter = SemanticChunker(GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    semantic_chunks = semantic_text_splitter.create_documents([text])

    semantic_vector_store = FAISS.from_documents(semantic_chunks, embeddings)
    semantic_retriever = semantic_vector_store.as_retriever()

    initial_prompt = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant specifically designed to work with RAG (Retrieval Augmented Generation) systems. "
                "Your responses must be strictly based on the provided context. Follow these rules:\n"
                "1. Only answer using information explicitly present in the given context\n"
                "2. If the context is not relevant to the query, respond with 'The provided context is not relevant to answer this query.'\n"
                "3. If the context is partially relevant but incomplete, respond with 'The context is incomplete to fully answer this query. "
                "Based on the available context, I can only tell you that: [insert relevant information from context]'\n"
                "4. Do not make assumptions or add information beyond what's in the context\n"
                "5. If you quote or reference specific parts of the context, indicate this in your response\n"
                "6. If the context is empty or missing, respond with 'No context was provided to answer this query.'\n\n"
                "Remember: You are a context-dependent system. Never draw from general knowledge - only use the specific context provided in each query."
            ),
        },
        {"role": "user", "content": query}
    ]
    initial_response = llm.invoke(initial_prompt).content
    refined_answer = initial_response

    for _ in range(max_refinement_rounds):

        retrieved_docs = semantic_retriever.get_relevant_documents(refined_answer)
        if not retrieved_docs:
            break

        more_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        refine_prompt = [
            {
                "role": "system",
                "content": (
                    f"Below is the initial answer to the user's query:\n\n"
                    f"Initial Answer: {refined_answer}\n\n"
                    "Here is additional context retrieved based on that answer:\n"
                    f"{more_context}\n\n"
                    "Please refine or correct the initial answer strictly using the additional context. "
                    "Remember the RAG rules and do not add any information not found in the context."
                ),
            },
            {"role": "user", "content": query},
        ]
        new_answer = llm.invoke(refine_prompt).content

        refined_answer = new_answer
    return refined_answer


def graph_rag_process(text, embeddings, query, llm):
    #for this part, I built a graph with only subject and objects, which means it's a undirected graph, and there's no relation description
    #between two nodes, the nodes is connected sccording to the cossimularity with the key called neighbors. Then I append the neighbors 
    # text as relevant_context for input and use chain for answer
    semantic_text_splitter = SemanticChunker(GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    semantic_chunks = semantic_text_splitter.create_documents([text])

    max_paths_per_chunk = 3

    graph = {}
    for i, chunk in enumerate(semantic_chunks):
        graph[i] = {
            'content': chunk.page_content,
            'embeddings': embeddings.embed_query(chunk.page_content),
            'neighbors': []
        }

    for i in range(len(semantic_chunks)):
        paths = 0
        for j in range(i + 1, len(semantic_chunks)):
            similarity = cosine_similarity([graph[i]['embeddings']], [graph[j]['embeddings']])[0][0]
            if similarity > 0.5 and paths < max_paths_per_chunk:  
                graph[i]['neighbors'].append(j)
                graph[j]['neighbors'].append(i)
                paths += 1

    query_embedding = embeddings.embed_query(query)
    similarities = {i: cosine_similarity([query_embedding], [graph[i]['embeddings']])[0][0] for i in graph}
    most_similar_index = max(similarities, key=similarities.get)

    relevant_indices = [most_similar_index]
    visited = set([most_similar_index])
    queue = [most_similar_index]

    while queue:
        current = queue.pop(0)
        for neighbor in graph[current]['neighbors']:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                relevant_indices.append(neighbor)

    relevant_context = [Document(page_content=graph[i]['content']) for i in relevant_indices]
    relevant_vector_store = FAISS.from_documents(relevant_context, embeddings)
    relevant_retriever = relevant_vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": relevant_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query)



# Mapping of RAG types to their processing functions
RAG_IMPLEMENTATIONS = {
    "Naive RAG": naive_rag_process,
    "RAG with Semantic Chunking": semantic_chunking_rag_process,
    "RAG with Re-ranker": reranker_rag_process,
    "Hybrid RAG": hybrid_rag_process,
    "Self RAG": self_rag_process,
    "Graph RAG": graph_rag_process
}

# RAG descriptions for UI
RAG_DESCRIPTIONS = {
    "Naive RAG": "Basic RAG implementation with simple text chunking and embedding-based retrieval.",
    "RAG with Semantic Chunking": "RAG with intelligent text chunking based on semantic meaning.",
    "RAG with Re-ranker": "RAG with an additional re-ranking step to improve retrieval relevance.",
    "Hybrid RAG": "Combines semantic search and reranking for enhanced retrieval.",
    "Self RAG": "Self-reflective RAG that can generate and verify its own context.",
    "Graph RAG": "Graph-based RAG that maintains relationships between chunks of text."
}