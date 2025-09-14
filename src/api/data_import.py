"""
API endpoints for data import operations
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import tempfile
import shutil
from pathlib import Path

from src.config.database import get_db
from src.models.base import DataSource, BaseResponse
from src.services.data_import import DataImportService, DataQueryService

router = APIRouter(prefix="/api/data")


@router.post("/import/quickbooks", response_model=BaseResponse)
async def import_quickbooks_data(
    file: UploadFile = File(...), db: AsyncSession = Depends(get_db)
):
    """
    Import QuickBooks P&L data from JSON file
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        # Import data
        service = DataImportService(db)
        result = await service.import_file(tmp_path, DataSource.QUICKBOOKS)

        return BaseResponse(
            success=True,
            message=f"Imported {result['imported_records']} records successfully",
            data=result,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        # Clean up temporary file
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/import/rootfi", response_model=BaseResponse)
async def import_rootfi_data(
    file: UploadFile = File(...), db: AsyncSession = Depends(get_db)
):
    """
    Import Rootfi financial data from JSON file
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        # Import data
        service = DataImportService(db)
        result = await service.import_file(tmp_path, DataSource.ROOTFI)

        return BaseResponse(
            success=True,
            message=f"Imported {result['imported_records']} records successfully",
            data=result,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        # Clean up temporary file
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/import/local", response_model=BaseResponse)
async def import_local_files(db: AsyncSession = Depends(get_db)):
    """
    Import data from local data directory (for testing)
    """
    data_dir = Path("data")
    results = []

    # Import QuickBooks data
    qb_file = data_dir / "data_set_1.json"
    if qb_file.exists():
        service = DataImportService(db)
        try:
            result = await service.import_file(str(qb_file), DataSource.QUICKBOOKS)
            results.append(
                {"source": "QuickBooks", "file": "data_set_1.json", "result": result}
            )
        except Exception as e:
            results.append(
                {"source": "QuickBooks", "file": "data_set_1.json", "error": str(e)}
            )

    # Import Rootfi data
    rf_file = data_dir / "data_set_2.json"
    if rf_file.exists():
        service = DataImportService(db)
        try:
            result = await service.import_file(str(rf_file), DataSource.ROOTFI)
            results.append(
                {"source": "Rootfi", "file": "data_set_2.json", "result": result}
            )
        except Exception as e:
            results.append(
                {"source": "Rootfi", "file": "data_set_2.json", "error": str(e)}
            )

    return BaseResponse(
        success=True,
        message=f"Processed {len(results)} files",
        data={"imports": results},
    )


@router.get("/summary", response_model=BaseResponse)
async def get_data_summary(
    company_id: Optional[int] = None, db: AsyncSession = Depends(get_db)
):
    """
    Get summary of imported financial data
    """
    service = DataQueryService(db)
    summary = await service.get_summary_stats(company_id)

    return BaseResponse(success=True, message="Data summary retrieved", data=summary)


@router.get("/companies", response_model=BaseResponse)
async def get_companies(db: AsyncSession = Depends(get_db)):
    """
    Get list of all companies
    """
    service = DataQueryService(db)
    companies = await service.get_companies()

    return BaseResponse(
        success=True,
        message=f"Found {len(companies)} companies",
        data=[
            {"id": c.id, "name": c.name, "created_at": c.created_at} for c in companies
        ],
    )
