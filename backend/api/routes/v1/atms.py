import logging
from datetime import datetime
from typing import List, Optional

from database import get_db
from fastapi import APIRouter, Depends, HTTPException, Query
from models import ATM
from schemas.atm import ATMCreate, ATMListResponse, ATMResponse, ATMUpdate
from services import BackgroundTaskService, CacheService, TelemetryService
from sqlalchemy import or_
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
router = APIRouter()

# Service instances (will be set during startup)
cache_service: CacheService = None
telemetry_service: TelemetryService = None
background_service: BackgroundTaskService = None


def set_services(
    cache_svc: CacheService,
    telemetry_svc: TelemetryService,
    background_svc: BackgroundTaskService,
):
    """Set service instances"""
    global cache_service, telemetry_service, background_service
    cache_service = cache_svc
    telemetry_service = telemetry_svc
    background_service = background_svc


@router.post("/atms", response_model=ATMResponse, status_code=201)
async def create_atm(atm_data: ATMCreate, db: Session = Depends(get_db)):
    """Create a new ATM"""
    try:
        # Check if ATM with same ID already exists
        existing_atm = db.query(ATM).filter(ATM.atm_id == atm_data.atm_id).first()
        if existing_atm:
            raise HTTPException(
                status_code=409, detail=f"ATM with ID {atm_data.atm_id} already exists"
            )

        new_atm = ATM(
            atm_id=atm_data.atm_id,
            name=atm_data.name,
            location_address=atm_data.location_address,
            model=atm_data.model,
            manufacturer=atm_data.manufacturer,
            status=atm_data.status or "active",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        db.add(new_atm)
        db.commit()
        db.refresh(new_atm)

        # Invalidate cache
        await cache_service.invalidate_pattern("atm_*")
        await cache_service.invalidate_pattern("dashboard_*")

        logger.info(f"Created new ATM: {new_atm.atm_id}")

        return ATMResponse(
            atm_id=new_atm.atm_id,
            name=new_atm.name,
            location_address=new_atm.location_address,
            model=new_atm.model,
            manufacturer=new_atm.manufacturer,
            status=new_atm.status,
            created_at=new_atm.created_at.isoformat(),
            updated_at=new_atm.updated_at.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating ATM: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/atms", response_model=ATMListResponse)
async def get_atms(
    page: int = Query(0, ge=0, description="Page number (0-based)"),
    limit: int = Query(10, ge=1, le=1000, description="Number of records to return"),
    status: Optional[str] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Search by name or location"),
    region: Optional[str] = Query(
        None, description="Filter by region (extracted from atm_id)"
    ),
    db: Session = Depends(get_db),
):
    """Get list of ATMs with filtering and pagination"""
    try:
        # Calculate skip based on page
        skip = page * limit

        # Build query
        query = db.query(ATM)

        # Apply filters
        if status:
            query = query.filter(ATM.status == status)

        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    ATM.name.ilike(search_term),
                    ATM.location_address.ilike(search_term),
                    ATM.atm_id.ilike(search_term),
                )
            )

        if region:
            query = query.filter(ATM.atm_id.like(f"%{region}%"))

        # Get total count
        total = query.count()

        # Apply pagination and get results
        atms = query.offset(skip).limit(limit).all()

        # Convert to response format
        atm_list = []
        for atm in atms:
            atm_list.append(
                ATMResponse(
                    atm_id=atm.atm_id,
                    name=atm.name,
                    location_address=atm.location_address,
                    model=atm.model,
                    manufacturer=atm.manufacturer,
                    status=atm.status,
                    created_at=atm.created_at.isoformat(),
                    updated_at=atm.updated_at.isoformat(),
                )
            )

        return ATMListResponse(data=atm_list, page=page, limit=limit, total=total)

    except Exception as e:
        logger.error(f"Error getting ATMs: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/atms/{atm_id}", response_model=ATMResponse)
async def get_atm(atm_id: str, db: Session = Depends(get_db)):
    """Get a specific ATM by ID"""
    try:
        # Try cache first
        cache_key = f"atm_details:{atm_id}"
        cached_atm = await cache_service.get(cache_key)
        if cached_atm:
            return ATMResponse(**cached_atm)

        # Get from database
        atm = db.query(ATM).filter(ATM.atm_id == atm_id).first()
        if not atm:
            raise HTTPException(status_code=404, detail=f"ATM {atm_id} not found")

        atm_response = ATMResponse(
            atm_id=atm.atm_id,
            name=atm.name,
            location_address=atm.location_address,
            model=atm.model,
            manufacturer=atm.manufacturer,
            status=atm.status,
            created_at=atm.created_at.isoformat(),
            updated_at=atm.updated_at.isoformat(),
        )

        # Cache the result
        await cache_service.set(cache_key, atm_response.dict(), ttl=300)

        return atm_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting ATM {atm_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/atms/{atm_id}", response_model=ATMResponse)
async def update_atm(atm_id: str, atm_data: ATMUpdate, db: Session = Depends(get_db)):
    """Update an existing ATM"""
    try:
        # Get existing ATM
        atm = db.query(ATM).filter(ATM.atm_id == atm_id).first()
        if not atm:
            raise HTTPException(status_code=404, detail=f"ATM {atm_id} not found")

        # Update fields that are provided
        update_data = atm_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(atm, field, value)

        atm.updated_at = datetime.now()

        db.commit()
        db.refresh(atm)

        # Invalidate cache
        await cache_service.delete(f"atm_details:{atm_id}")
        await cache_service.invalidate_pattern("atm_*")
        await cache_service.invalidate_pattern("dashboard_*")

        logger.info(f"Updated ATM: {atm_id}")

        return ATMResponse(
            atm_id=atm.atm_id,
            name=atm.name,
            location_address=atm.location_address,
            model=atm.model,
            manufacturer=atm.manufacturer,
            status=atm.status,
            created_at=atm.created_at.isoformat(),
            updated_at=atm.updated_at.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating ATM {atm_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
