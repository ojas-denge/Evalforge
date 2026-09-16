from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from app.db.repository import EvaluationRepository

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.models.schemas import QueryRequest, QueryResponse
from app.retrieval.retriever import Retriever
from app.observability.tracing import Tracer


settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

tracer = Tracer()
retriever = Retriever(tracer=tracer)
evaluation_repository = EvaluationRepository()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting %s v%s in %s environment",
        settings.app_name,
        settings.app_version,
        settings.app_env,
    )
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    description="LLM Evaluation & Observability Platform",
    version=settings.app_version,
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    logger.info("Health check requested")
    return {
        "status": "ok",
        "service": settings.app_name.lower(),
        "version": settings.app_version,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    with tracer.trace(
        name="rag_request",
        input={"question": request.question},
        metadata={
            "endpoint": "/query",
            "retrieval_top_k": 3,
        },
    ) as observation:

        logger.info(
            "Query received: %s",
            request.question,
        )

        retrieval = retriever.retrieve(
            query=request.question,
            top_k=3,
        )

        citations = [
            result.document_id
            for result in retrieval.results
        ]

        response = QueryResponse(
            answer="Retrieval successful. Generation not implemented yet.",
            citations=citations,
            confidence=0.0,
            latency_ms=retrieval.latency_ms,
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=0.0,
        )

        if observation is not None:
            observation.update(
                output={
                    "citations": citations,
                    "retrieval_result_count": len(retrieval.results),
                },
            )

        return response

@app.get("/evaluations")
def list_evaluations():
    return evaluation_repository.list_runs()


@app.get("/evaluations/{run_id}")
def get_evaluation(run_id: str):
    run = evaluation_repository.get_run(run_id)

    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation run not found: {run_id}",
        )

    return run