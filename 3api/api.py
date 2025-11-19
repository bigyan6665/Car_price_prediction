from fastapi import FastAPI
import pickle as pk, pandas as pd
import os

app = FastAPI()


@app.get("/")
def home():
    return {"Hello": "World"}


@app.get("/predict/")
def prediction(
    Brand: str,
    Body: str,
    Mileage: float,
    EngineV: float,
    EngineType: str,
    Registration: str,
    Year: int,
    Model: str,
):
    path = os.path.join(os.path.dirname(__file__), "car_price_predictor.pickle")
    with open(path, "rb") as f:
        model = pk.load(f)

    df = pd.DataFrame(
        {
            "Brand": [Brand],
            "Body": [Body],
            "Mileage": [Mileage],
            "EngineV": [EngineV],
            "Engine Type": [EngineType],
            "Registration": [Registration],
            "Year": [Year],
            "Model": [Model],
        }
    )
    # print(round(model.predict(df)[0][0], 2))
    return {"Predicted_price": round(model.predict(df)[0][0], 2)}
