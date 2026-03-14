from fastapi import APIRouter
from app.schemas import PredictRequest, PredictResponse, TaskStatusResponse
from app.worker.tasks import predict_anomaly_task
from app.worker.celery_app import celery_app

router = APIRouter()

@router.post("/predict", response_model=PredictResponse, status_code=202)
def predict_anomaly(request: PredictRequest):
    """
    Predict anomaly score based on incoming financial time-series data.
    
    Time Complexity (Big-O):
    - Data Validation (Pydantic): O(N) where N is the number of features (20 in this case).
      It iterates through the list to validate type and length.
    - Redis Task Offloading: O(1) constant time, extremely fast memory push.
    - Overall API Route Complexity: O(N) dominated by JSON parsing and validation.
    
    Quantum CPU/QPU execution is completely decoupled.
    """
    
    # Dispatch the Celery task
    task = predict_anomaly_task.delay(request.features)
    
    # Return immediately avoiding blocked threads
    return PredictResponse(task_id=task.id)

@router.get("/status/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """Check the processing status of a dispatched Celery job from Redis."""
    
    task_result = celery_app.AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,
        "result": None
    }
    
    if task_result.status == 'SUCCESS':
        response["result"] = task_result.result
        
    return TaskStatusResponse(**response)
