from io import BytesIO
from turtle import st

from fastapi import APIRouter, Body, Depends, File, HTTPException, Path, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

from app.resources.CustomLogger import CustomLogger
from app.resources.RedisFileProcessor import RedisFileProcessor

logger = CustomLogger(__name__).get_logger()
router = APIRouter()

# Prometheus metrics
upload_counter = Counter("file_uploads_total", "Total number of file uploads")
upload_errors_counter = Counter("file_upload_errors_total", "Total number of file upload errors")
queue_errors_counter = Counter("queue_errors_total", "Total number of queue retrieval errors")

###
# TODO: Add authentication and authorization, use dependencies module of APIRouter for that
###


@router.post("/upload")
async def upload_file(file: UploadFile):
    """
    Upload a file endpoint.

    Example curl command:
    curl -X POST "http://localhost:8000/upload" -F "file=@/path/to/your/file.txt"
    """
    try:
        contents = await file.read()
        # Process the file contents here
        logger.info(f"File upload: {file.filename}, size: {len(contents)} bytes")

        file_like_object = BytesIO(contents)

        RedisObject = RedisFileProcessor()
        # Upload the file with notification
        await RedisObject.upload_file_with_notif(
            file_object=file_like_object,
            filename=file.filename or "unknown_file",  # Assuming the file is saved locally first
            metadata={"filename": file.filename, "size": len(contents)},
        )

        # Increment successful upload counter
        upload_counter.inc()

        return {"filename": file.filename, "size": len(contents)}
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}", exc_info=True, stack_info=True)

        # Increment error counter
        upload_errors_counter.inc()

        raise HTTPException(status_code=500, detail="File upload failed") from e


@router.get("/queue")
async def get_queue():
    """
    Get the current state of the file upload queue.
    """
    try:
        RedisObject = RedisFileProcessor()
        queue_length = await RedisObject.get_queue_status()
        return {"queue_length": queue_length}
    except Exception as e:
        logger.error(f"Error retrieving queue: {str(e)}", exc_info=True, stack_info=True)

        # Increment error counter
        queue_errors_counter.inc()

        raise HTTPException(status_code=500, detail="Failed to retrieve queue") from e


@router.get("/metrics")
async def get_metrics():
    """
    Prometheus metrics endpoint.
    """
    return generate_latest()
