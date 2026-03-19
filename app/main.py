from fastapi import FastAPI
import joblib
import numpy as np

from prometheus_client import Counter, generate_latest
from starlette.responses import Response

app = FastAPI()

model = joblib.load("model/iris_model.pkl")

REQUEST_COUNT = Counter("prediction_requests_total", "Total prediction requests")

@app.get("/")
def home():
    return {"message": "ML API Running"}

@app.post("/predict")
def predict(sepal_length: float,
            sepal_width: float,
            petal_length: float,
            petal_width: float):

    REQUEST_COUNT.inc()

    data = np.array([[sepal_length,
                      sepal_width,
                      petal_length,
                      petal_width]])

    prediction = model.predict(data)

    return {"prediction": int(prediction[0])}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
# trigger ci cd