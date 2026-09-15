\# Context Management



Retrieved evidence is passed to the generation layer as model context.



More retrieved context is not always better.



Additional context can improve answer completeness when relevant evidence is distributed across multiple chunks.



Irrelevant context can increase generation cost and make it harder for a model to identify the most useful evidence.



Context construction should therefore balance evidence coverage, relevance, context length, latency, and cost.

