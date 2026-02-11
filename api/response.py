from fastapi import HTTPException
from fastapi.responses import JSONResponse


def success_response(message: str, data: dict):
    return {
        "success": True,
        "message": message,
        "data": data
    }

def error_response(message: str, errors=None, status_code: int = 400):
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "message": message,
            "errors": errors or []
        }
    )