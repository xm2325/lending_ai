from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class CreditSequenceRequest(BaseModel):
    sequence: list[list[float]] = Field(min_length=1, max_length=6)
    static: list[float] = Field(min_length=3, max_length=3)

    @field_validator("sequence")
    @classmethod
    def validate_sequence_width(cls, value: list[list[float]]) -> list[list[float]]:
        if any(len(row) != 5 for row in value):
            raise ValueError("Each monthly row must contain five sequence features")
        return value


class CreditRiskResponse(BaseModel):
    risk_score: float
    model_version: str
    decision_use: str = "research_demo_only"
