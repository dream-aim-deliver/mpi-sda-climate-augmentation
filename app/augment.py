from logging import Logger
import logging
from typing import List
from app.sdk.models import KernelPlancksterSourceData, BaseJobState, JobOutput, ProtocolEnum
from app.sdk.scraped_data_repository import ScrapedDataRepository,  KernelPlancksterSourceData
from app.weather_API import *
import os
import json
import pandas as pd
from dotenv import load_dotenv
from models import PipelineRequestModel
from numpy import ndarray
import numpy as np
import shutil
import cv2
import tempfile
import torch
import torchvision.transforms as transforms
from collections import Counter
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import time
from PIL import Image

def augment(
    job_id: int,
    tracer_id: str,
    scraped_data_repository: ScrapedDataRepository,
    log_level: Logger,
    work_dir: str,
    protocol:str,


) -> JobOutput:


    try:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=log_level)

        job_state = BaseJobState.CREATED
        protocol = scraped_data_repository.protocol
    
        # Set the job state to running
        logger.info(f"{job_id}: Starting Job")
        job_state = BaseJobState.RUNNING
        #job.touch()

        start_time = time.time()  # Record start time for response time measurement 
        #Download all relevant files from minio
        kernel_planckster = scraped_data_repository.kernel_planckster
        source_check = False   #make True if source list needs to be checked
        # Get the source list
        source_list = kernel_planckster.list_all_source_data()
        if source_check:
            output_file = "source_list.txt"
            with open(output_file, 'w') as file:
                file.write("\n".join(map(str, source_list)))
            print(f"Source list saved to {output_file}")      #to check the relative paths returned by MinIO
        


        minimum_info = {"sentinel": False, "Webcam":False}
        try:
            for source in source_list:
                    res = download_source_if_relevant(source, job_id, tracer_id, log_level, scraped_data_repository, work_dir) 
                    if res["sentinel"] == True: minimum_info["sentinel"] = True
                    elif res["Webcam"] == True: minimum_info["Webcam"] = True; 
                    logger.info("Successfully downloaded")
        except Exception as e:
                logger.warning(f"Download error : {e}")
        
    
        if minimum_info["sentinel"] == True and minimum_info["Webcam"]==True:
            try:
                augment_by_date(work_dir, job_id, scraped_data_repository, protocol)
                logger.debug("Sentinel data augmented successfully")
                augment_image(job_id, scraped_data_repository, log_level, work_dir, protocol)
                logger.debug("Webcam images augmented successfully")
                job_state = BaseJobState.FINISHED
                shutil.rmtree(work_dir)
                logger.info(f"Augmentation complete, successfully deleted {work_dir} directory")
            except Exception as e:
                logger.error(f"Could not augment data. Error:{e}")   
        else:
            logger.warning("Could not run augmentation, try again after running data pipeline for sentinel and webcam")
            try:
                shutil.rmtree(work_dir)
                logger.info(f"successfully deleted {work_dir} directory")
            except Exception as e:
                logger.warning(f"Could not delete tmp directory due to {e}, exiting")

            

                
        
    except Exception as error:
        logger.error(f"Job {job_id}, Tracer {tracer_id}: Job failed due to {error}")
        job_state = BaseJobState.FAILED
        try:
            shutil.rmtree(work_dir)
            print(f"successfully deleted {work_dir} directory")
        except Exception as e:
            logger.warning(f"Could not delete tmp directory due to {e}, exiting")


def download_source_if_relevant(
    source: KernelPlancksterSourceData,
    job_id: int,
    tracer_id: str,
    log_level: str,
    scraped_data_repository: ScrapedDataRepository,
    work_dir: str
):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=log_level)
    name = source["name"]
    protocol = source["protocol"]
    relative_path = source["relative_path"]

    # Reconstruct the source_data object
    source_data = KernelPlancksterSourceData(
        name=name,
        protocol=protocol,
        relative_path=relative_path,
    )
    file_name = os.path.basename(relative_path)

    res = {"sentinel": False, "Webcam": False}
    # Handle JSON downloads from Sentinel
    if "climate" in source_data.relative_path and os.path.splitext(source_data.relative_path)[1] == ".json":
        sentinel_coords_path = os.path.join(work_dir, "climate_coords", file_name)
        try:
            scraped_data_repository.download_json(source_data, job_id, sentinel_coords_path)
            res["sentinel"] = True
            logger.debug("sentinel data download success")
        except FileNotFoundError:
            logger.error(f"File not found in MinIO: {relative_path}")
            return res
    
    elif "Webcam/1/1/scraped" in source_data.relative_path:
        image_save_path = os.path.join(work_dir, "Webcam", "scraped_images", file_name)
        try:
            scraped_data_repository.download_image(source_data, job_id, image_save_path)
            res["Webcam"] = True
            logger.debug(f"Downloaded image to {image_save_path}")
        except FileNotFoundError:
            logger.error(f"Image file not found in MinIO: {relative_path}")
            return res

    return res


def augment_by_date(work_dir: str, job_id: int, scraped_data_repository: ScrapedDataRepository, protocol: ProtocolEnum):
    key = {
        "01": "January",
        "02": "February",
        "03": "March",
        "04": "April",
        "05": "May",
        "06": "June",
        "07": "July",
        "08": "August",
        "09": "September",
        "10": "October",
        "11": "November",
        "12": "December"
    }

    sentinel_dir = os.path.join(work_dir, "climate_coords")
    
    for climate_coords_json_file_path in os.listdir(sentinel_dir):
        data = []
        sentinel_df = pd.read_json(os.path.join(sentinel_dir, climate_coords_json_file_path), orient="index")
        
        for index, row in sentinel_df.iterrows():
            lattitude = row.get('latitude')
            longitude = row.get('longitude')
            CO_level = row.get('CO_level')
            data.append([CO_level, lattitude, longitude])

        try:
            # Extract date part from the file name by finding content between the first and second "__"
            start_date_part = climate_coords_json_file_path.split("__")[1]  # Extracts '2023_12_13' part
            split_date = start_date_part.split("_")

            if len(split_date) != 3:
                raise ValueError(f"Invalid date format in file name: {climate_coords_json_file_path}")

            sat_image_year = split_date[0]
            sat_image_month = key.get(split_date[1])
            sat_image_day = split_date[2]

            if not sat_image_month:
                raise ValueError(f"Invalid month in file name: {climate_coords_json_file_path}")

            # Create DataFrame and save to JSON
            date_df = pd.DataFrame(data, columns=["CO_level", "Latitude", "Longitude"])
            os.makedirs(f"{work_dir}/by_date", exist_ok=True)
            local_json_path = f"{work_dir}/by_date/{sat_image_year}_{sat_image_month}_{sat_image_day}.json"
            date_df.to_json(local_json_path, orient='index', indent=4)
            
            # Upload to MinIO
            source_data = KernelPlancksterSourceData(
                name=f"{sat_image_year}_{sat_image_month}_{sat_image_day}",
                protocol=protocol,
                relative_path=f"augmented/by_date/{sat_image_year}_{sat_image_month}_{sat_image_day}.json"
            )

            try:
                scraped_data_repository.register_scraped_json(
                    job_id=job_id,
                    source_data=source_data,
                    local_file_name=local_json_path,
                )
            except Exception as e:
                logging.error(f"Failed to register scraped JSON for {local_json_path}: {str(e)}")
                continue

        except ValueError as ve:
            logging.error(str(ve))
            continue
        except Exception as e:
            logging.error(f"Error processing file {climate_coords_json_file_path}: {str(e)}")
            continue

def augment_image(job_id, scraped_data_repository, log_level, work_dir, protocol):
    # Configure logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=log_level)
    
    webcam_dir = os.path.join(work_dir, "Webcam", "scraped_images")
    combined_dir = os.path.join(work_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)
    
    if not os.path.exists(webcam_dir):
        logger.error(f"Webcam directory not found: {webcam_dir}")
        return

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.eval()
    
    for image_file in os.listdir(webcam_dir):
        try:
            image_path = os.path.join(webcam_dir, image_file)
            date_from_filename = extract_date_from_filename(image_file)
            latitude, longitude = extract_latitude_longitude_from_filename(image_file)
            formatted_date = format_date(date_from_filename)
            split_date = formatted_date.split("-")
            
            # Classify weather from the image
            majority_weather_from_image = process_and_classify_weather(image_path, model)
            
            # Fetch and process weather data from API
            response = fetch_weather_data(latitude=latitude, longitude=longitude,
                                        start_date=formatted_date, end_date=formatted_date)
            weather_condition = None
            if response:
                weather_data = process_weather_data(response)
                if not weather_data.empty:
                    weather_condition = get_weather_condition(weather_data)
            
            combined_filename = f"{split_date[0]}_{split_date[1]}_{split_date[2]}.json"
            local_json_path = os.path.join(combined_dir, combined_filename)
            save_results_to_json(
                url=None,
                date=date_from_filename,
                latitude=latitude,
                longitude=longitude,
                weather={"from_image": majority_weather_from_image, "from_api": weather_condition},
                filename=local_json_path
            )
            
            if os.path.exists(local_json_path):
                logger.info(f"File {local_json_path} created successfully.")
            else:
                logger.error(f"Failed to create file {local_json_path}")
                continue
            
            source_data = KernelPlancksterSourceData(
                name=combined_filename,
                protocol=protocol,
                relative_path=f"augmented/combined/{combined_filename}"
            )
            scraped_data_repository.register_scraped_json(
                job_id=job_id,
                source_data=source_data,
                local_file_name=local_json_path,
            )
        except Exception as e:
            logger.error(f"Failed to process image {image_file}: {e}")
            continue
