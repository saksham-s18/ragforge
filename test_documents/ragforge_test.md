\# RAGForge Test Knowledge Base



\## Project Overview



RAGForge is a production-oriented Retrieval-Augmented Generation system.



The project uses FastAPI as its API framework, Qdrant as its vector database,

and FastEmbed for production document embeddings.



\## Retrieval



RAGForge converts documents into chunks and generates embeddings for those chunks.

The embeddings are stored in Qdrant.



When a user asks a question, the question is embedded and used to retrieve

the most relevant document chunks from Qdrant.



\## LLM Generation



RAGForge uses Groq as its primary LLM provider.



OpenAI is configured as a fallback provider when Groq encounters a

fallback-eligible transient failure such as a timeout, rate limit,

or temporary provider unavailability.



\## Security



API keys are stored in environment variables and must never be committed

to the Git repository.



RAGForge does not allow retrieved document content to override the

system instructions used for grounded generation.



\## Testing



RAGForge uses pytest for automated testing, Ruff for linting and formatting,

and mypy for static type checking.

