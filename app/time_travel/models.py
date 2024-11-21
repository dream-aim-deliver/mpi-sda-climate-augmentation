from pydantic import BaseModel
from typing import List, Union

class ClimateRowSchema(BaseModel):
    timestamp: str
    latitude: float
    longitude: float
    CarbonMonoxideLevel: str
    PredictedWeather: str
    ActualWeather: str


class Error(BaseModel):
    errorName: str
    errorMessage: str

class Image(BaseModel):
    kind: str
    relativePath: str
    description: str


class KeyFrame(BaseModel):
    timestamp: str
    images: List[Union[Image, Error]]
    data: List[Union[ClimateRowSchema, Error]]
    dataDescription: str

class Metadata(BaseModel):
    caseStudy: str
    relativePathsForAgent: List[str]
    keyframes: List[KeyFrame]
    imageKinds: List[str]
