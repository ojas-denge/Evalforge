# Retrieval Guidelines

The baseline retrieval implementation uses semantic similarity search.

Documents are divided into chunks before they are embedded.

Each chunk should contain enough surrounding context to preserve meaning, while avoiding unnecessarily large chunks.

The baseline system retrieves a configurable number of top-k chunks.

Increasing top-k can improve recall when relevant information is distributed across multiple chunks, but excessive context can introduce irrelevant information and increase generation cost.

Retrieval quality should be evaluated independently from final answer quality.

A retrieval failure occurs when the information required to answer a question is absent from the retrieved context even though it exists in the knowledge base.

The baseline retrieval system should be measured before introducing hybrid retrieval or reranking.
