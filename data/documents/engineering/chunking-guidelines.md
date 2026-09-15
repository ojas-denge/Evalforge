\# Chunking Guidelines



Document chunking determines the units that are indexed and retrieved by the vector store.



Chunks should preserve enough surrounding context for their content to remain meaningful when retrieved independently.



Very large chunks can contain unrelated information and reduce retrieval precision.



Very small chunks can remove important context and make evidence difficult to interpret.



Chunk size and overlap should therefore be treated as retrieval parameters that can be evaluated experimentally.



Changes to chunking can alter retrieval rankings even when the embedding model remains unchanged.

