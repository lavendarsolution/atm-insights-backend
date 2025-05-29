from typing import Generic, List, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Common paginated response schema"""

    data: List[T]
    page: int = Field(..., description="Current page number", ge=0)
    limit: int = Field(..., description="Number of records per page")
    total: int = Field(..., description="Total number of records")

    class Config:
        from_attributes = True
