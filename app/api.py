from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from redis import Redis
from rq import Queue
from app.tasks import process_image_task
from app.detector import PlateDetector
from app.analyzer import PlateAnalyzer

MODEL_PATH = "runs/plate_yolo/weights/best.pt"
redis_conn = Redis(host="localhost", port=6379)
queue = Queue("plates", connection=redis_conn)
app = FastAPI()

detector = PlateDetector(MODEL_PATH)
analyzer = PlateAnalyzer(detector)


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    data = await file.read()
    img_array = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    result = analyzer.analyze(image)
    return result


@app.post("/enqueue")
async def enqueue_image(file: UploadFile = File(...)):
    data = await file.read()
    job = queue.enqueue(process_image_task, data)
    return {"job_id": job.id}