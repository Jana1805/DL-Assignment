"""HTTP API for Just-In-Time defect prediction on Apache commits."""

import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from jit.artifacts import load_artifacts
from jit.features import RAW_FEATURES

logger = logging.getLogger("jit.api")


class CommitMetrics(BaseModel):
    """The 12 structural change metrics describing one commit."""

    model_config = ConfigDict(extra="forbid")

    la: int = Field(ge=0, description="Lines added")
    ld: int = Field(ge=0, description="Lines deleted")
    nf: int = Field(ge=1, description="Files touched")
    nd: int = Field(ge=1, description="Directories touched")
    ns: int = Field(ge=1, description="Subsystems touched")
    ent: float = Field(ge=0, allow_inf_nan=False, description="Change entropy")
    ndev: float = Field(ge=0, allow_inf_nan=False, description="Distinct prior developers")
    age: float = Field(ge=0, allow_inf_nan=False, description="Avg. time since last change")
    nuc: float = Field(ge=0, allow_inf_nan=False, description="Unique prior changes")
    aexp: float = Field(ge=0, allow_inf_nan=False, description="Author experience")
    arexp: float = Field(ge=0, allow_inf_nan=False, description="Author recent experience")
    asexp: float = Field(ge=0, allow_inf_nan=False, description="Author subsystem experience")


class Prediction(BaseModel):
    risk_score: float = Field(description="Estimated probability the commit is bug-inducing")
    is_risky: bool = Field(description="risk_score at or above the tuned threshold")
    threshold: float
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.environ.get("MODEL_DIR", "artifacts"))
    app.state.model, app.state.metadata = load_artifacts(model_dir)
    logger.info("Loaded model %s", app.state.metadata["model_version"])
    yield


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Enforce the X-API-Key header when the API_KEY variable is set."""
    expected = os.environ.get("API_KEY")
    if expected and not secrets.compare_digest((x_api_key or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def create_app() -> FastAPI:
    app = FastAPI(
        title="JIT Defect Prediction API",
        description=(
            "Scores how likely a commit is to introduce a bug. Intended as a "
            "review hint, not a merge gate: expect many false alarms."
        ),
        lifespan=lifespan,
    )

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/health")
    def health(request: Request) -> dict:
        return {
            "status": "ok",
            "model_version": request.app.state.metadata["model_version"],
        }

    @app.post("/predict", response_model=Prediction, dependencies=[Depends(require_api_key)])
    def predict(commit: CommitMetrics, request: Request) -> Prediction:
        frame = pd.DataFrame([commit.model_dump()], columns=RAW_FEATURES)
        score = float(request.app.state.model.score(frame)[0])
        metadata = request.app.state.metadata
        return Prediction(
            risk_score=round(score, 4),
            is_risky=score >= metadata["threshold"],
            threshold=metadata["threshold"],
            model_version=metadata["model_version"],
        )

    return app


app = create_app()
