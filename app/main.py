from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import EntityMatcher

app = FastAPI()
# models folder contains config.json file and model.safetensors file
path_to_saved_model = 'models'
model = EntityMatcher(model_path=path_to_saved_model)


class EntityPair(BaseModel):
    entity_1: str
    entity_2: str


@app.post("/predict")
def predict_similarity(pair: EntityPair):

    prediction, probability = model.predict(pair.entity_1, pair.entity_2)
    return {"prediction": prediction, "probability": round(probability, 6)}
