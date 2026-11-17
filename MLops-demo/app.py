import mlflow
import mlflow.sklearn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Iris MLOps Level 2 API")

# Same database as retrain_pipeline.py
mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "IrisRandomForestModel"

model = None


def load_latest_model():

    try:
        model_uri = f"models:/{MODEL_NAME}@champion"

        loaded_model = mlflow.sklearn.load_model(model_uri)

        print("Champion model loaded successfully!")

        return loaded_model

    except Exception as e:

        print("MLflow model loading failed:", str(e))

        return None


@app.on_event("startup")
def startup_event():

    global model
    model = load_latest_model()
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }



class IrisInput(BaseModel):

    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/")
def home():

    return {"message": "Iris Classification API is running"}


@app.get("/health")
def health():

    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(data: IrisInput):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model artifact unavailable."
        )

    try:

        features = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]]

        prediction = model.predict(features)

        classes = [
            "setosa",
            "versicolor",
            "virginica"
        ]

        result = int(prediction[0])

        return {
            "prediction": result,
            "flower_name": classes[result]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )