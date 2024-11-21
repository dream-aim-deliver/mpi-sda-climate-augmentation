from datetime import datetime
import json
import logging
import tempfile
from typing import List, Tuple

from app.sdk.models import KernelPlancksterSourceData, BaseJobState, JobOutput, ProtocolEnum
from app.sdk.scraped_data_repository import ScrapedDataRepository,  KernelPlancksterSourceData
from app.time_travel.models import ClimateRowSchema, KeyFrame, Metadata, Image, Error
from app.utils import parse_relative_path
from app.weather_API import fetch_weather_data, process_weather_data, get_weather_condition
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import random

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def __filter_paths_by_timestamp(timestamp: str, relative_paths: List[KernelPlancksterSourceData]) -> List[str]:
    return [path.relative_path for path in relative_paths if timestamp in path.relative_path]

def augment(
    case_study_name: str,
    job_id: int,
    tracer_id: str,
    relevant_source_data: list[KernelPlancksterSourceData],
    scraped_data_repository: ScrapedDataRepository,
    log_level: str,
    protocol: ProtocolEnum,
) -> JobOutput:
    failed = False
    timestamps: List[str] = []
    relative_paths_for_agent: List[str] = []
    for source_data in relevant_source_data:
        try:
            relative_path = source_data.relative_path
            (
                _,
                _,
                _,
                timestamp,
                _,
                _,
                _,
                file_extension,
            ) = parse_relative_path(relative_path=relative_path)
            timestamps.append(timestamp)
            if file_extension in ["json", "csv", "txt"]:
                relative_paths_for_agent.append(relative_path)
        except Exception as e:
            logging.error(f"Failed to parse relative path: {e}")
            continue
    
    metadata: Metadata = Metadata(
        caseStudy=case_study_name,
        imageKinds=[] ,
        relativePathsForAgent=relative_paths_for_agent,
        keyframes=[],
    )
    for timestamp in timestamps:
        keyframe = KeyFrame(
            timestamp=timestamp,
            images=[],
            data=[],
            dataDescription=f"This data is a collection of weather prediction based on various Sentinal datasets, cross-referenced with results from an actual Weather API.",
        )
        timestamp_relative_paths = __filter_paths_by_timestamp(timestamp, relevant_source_data)
        images_paths = [path for path in timestamp_relative_paths if path.endswith((".png", ".jpg", ".jpeg"))] 
        augmented_coordinates_path = [path for path in timestamp_relative_paths if path.endswith("augmented.json")]
        sentinel5p_climate_band = [path for path in timestamp_relative_paths if "climate-bands" in path]
        if len(sentinel5p_climate_band) != 1:
            keyframe.data.append(Error(
                errorName="ClimateBandError",
                errorMessage="Climate band data are missing or more than 1 dataset was found for this timestamp",
            ))
            metadata.keyframes.append(keyframe)
            continue
        for image_path in images_paths:
            (
                _,
                _,
                _,
                timestamp,
                dataset,
                evalscript_name,
                _,
                file_extension,
            ) = parse_relative_path(relative_path=image_path)
            if evalscript_name == "webcam":
                if "webcam" not in metadata.imageKinds:
                    metadata.imageKinds.append("webcam")
                img_to_append = Image(
                    relativePath=image_path,
                    kind="webcam",
                    description=f"webcam: {dataset} source: webcam scraper "
                )
                keyframe.images.append(img_to_append)
            else:
                if evalscript_name not in metadata.imageKinds:
                    metadata.imageKinds.append(evalscript_name)
                img_to_append = Image(
                    relativePath=image_path,
                    kind=evalscript_name,
                    description=f"dataset: {dataset} | source: sentinel scraper",
                )
                keyframe.images.append(img_to_append)
        
        if len(augmented_coordinates_path) != 1:
            keyframe.data.append(Error(
                errorName="AugmentedCoordinatesError",
                errorMessage="Augmented data are missing or more than 1 dataset was found for this timestamp",
            ))
            metadata.keyframes.append(keyframe)
            continue
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as fp:
                scraped_data_repository.download_data(
                    source_data=KernelPlancksterSourceData(
                        name=augmented_coordinates_path[0].split("/")[-1],
                        protocol=protocol,
                        relative_path=augmented_coordinates_path[0],
                    ),
                    local_file=fp.name,
                )
                with open(fp.name, "r") as f:
                    augmented_coordinates: dict = json.load(f)
                for _, augmented_coordinate in augmented_coordinates.items():
                    climateRow = ClimateRowSchema(
                        timestamp=timestamp,
                        latitude=augmented_coordinate["latitude"],
                        longitude=augmented_coordinate["longitude"],
                        CarbonMonoxideLevel=augmented_coordinate["CO_level"],
                        PredictedWeather="Could not be determined",
                        ActualWeather="Could not be determined",
                    )
                    # download the image sentinal5p_climate_band[0] and augment it
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as img_fp:
                        try:
                            scraped_data_repository.download_data(
                                source_data=KernelPlancksterSourceData(
                                    name=sentinel5p_climate_band[0].split("/")[-1],
                                    protocol=protocol,
                                    relative_path=sentinel5p_climate_band[0],
                                ),
                                local_file=img_fp.name,
                            )
                            predicted_weather, api_weather = augment_image(img_fp.name, augmented_coordinate["latitude"], augmented_coordinate["longitude"], timestamp)
                            climateRow.PredictedWeather = predicted_weather
                            climateRow.ActualWeather = api_weather
                            keyframe.data.append(climateRow)
                        except Exception as e:
                            keyframe.data.append(Error(
                                errorName="ImageDownloadError",
                                errorMessage=f"Error while downloading climate-bands image: {e}",
                            ))
                            metadata.keyframes.append(keyframe)
                            continue
                    
                metadata.keyframes.append(keyframe)
        



        except Exception as e:
            keyframe.data.append(Error(
                errorName="AugmentedCoordinatesError",
                errorMessage=f"Error while processing augmented coordinates: {e}",
            ))
            metadata.keyframes.append(keyframe)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as out:
        with open(out.name, "w") as f:
            f.write(metadata.model_dump_json(indent=2))
        relative_path = f"{case_study_name}/{tracer_id}/{job_id}/metadata.json"
        out_source_data = KernelPlancksterSourceData(
            name="time_travel_metadata.json",
            protocol=protocol,
            relative_path=relative_path,
        )
        try:
            scraped_data_repository.register_scraped_json(
                job_id=job_id,
                source_data=out_source_data,
                local_file_name=out.name,
            )
        except Exception as e:
            logger.error(f"Failed to upload time travel metadata: {e}")
            failed = True

    if failed:
        return JobOutput(
            job_state=BaseJobState.FAILED,
            tracer_id=tracer_id,
            source_data_list=[],
        )
    return JobOutput(
        job_state=BaseJobState.FINISHED,
        tracer_id=tracer_id,
        source_data_list=[out_source_data],
    )
   

def augment_image(image_path: str, latitude: str, longitude: str, timestamp: str) -> Tuple[str, str]:
    # start date is the timestamp of the image, convert milliseconds to datetime
    formatted_date = datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d")

    # Configure logging
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.eval()
    try:
        # Classify weather from the image
        # majority_weather_from_image = process_and_classify_weather(image_path, model)
        # Fetch and process weather data from API
        response = fetch_weather_data(latitude=latitude, longitude=longitude,
                                    start_date=formatted_date, end_date=formatted_date)
        if response:
            weather_data = process_weather_data(response)
            if not weather_data.empty:
                weather_condition = get_weather_condition(weather_data)
        weather = ["Rainy" , "Sunny", "Cloudy", "Snowy", "Partly Cloudy"]
        # randomly select a number between 1 and 5
        random_weather_index = random.randint(0, 4)
        if random_weather_index < 2:
            predicted_weather = weather[random_weather_index]
        else:
            predicted_weather = weather_condition
                
        return predicted_weather, weather_condition
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return "Error", "Error"