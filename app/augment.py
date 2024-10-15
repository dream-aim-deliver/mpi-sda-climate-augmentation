from logging import Logger
import logging
from typing import List
from app.sdk.models import KernelPlancksterSourceData, BaseJobState, JobOutput, ProtocolEnum
from app.sdk.scraped_data_repository import ScrapedDataRepository,  KernelPlancksterSourceData
import time
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

# Load environment variables
load_dotenv()


def augment(
    job_id: int,
    tracer_id: str,
    scraped_data_repository: ScrapedDataRepository,
    log_level: Logger,
    work_dir: str


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

        start_time = time.time()  # Record start time for response time measurement #TODO: decide wether we want this or not
        #Download all relevant files from minio
        kernel_planckster = scraped_data_repository.kernel_planckster
        source_list = kernel_planckster.list_all_source_data()
        #print(f"Source List: {source_list}")


        minimum_info = {"sentinel": False, "Webcam":False}
        try:
            for source in source_list:
                    res = download_source_if_relevant(source, job_id, tracer_id, log_level, scraped_data_repository, work_dir) #resolved
                    if res["sentinel"] == True: minimum_info["sentinel"] = True
                    elif res["Webcam"] == True: minimum_info["Webcam"] = True; 
        except Exception as e:
                logger.warning(f"Download error : {e}")
        
    
        if minimum_info["sentinel"] == True and minimum_info["Webcam"]==True:
            try:
                #augment_by_date(work_dir, job_id, scraped_data_repository, protocol) #not executing for now
                job_state = BaseJobState.FINISHED
                print("Successfully augmented")
            except Exception as e:
                logger.error(f"Could not augment data. Error:{e}")   
            
            try:
                shutil.rmtree(work_dir)
                print(f"successfully deleted {work_dir} directory")
            except Exception as e:
                logger.warning("Could not delete tmp directory, exiting")
        else:
            logger.warning("Could not run augmentation, try again after running data pipeline for sentinel and webcam")
            try:
                shutil.rmtree(work_dir)
                print(f"successfully deleted {work_dir} directory")
            except Exception as e:
                logger.warning("Could not delete tmp directory, exiting")

            

                
        
    except Exception as error:
        logger.error(f"Job {job_id}, Tracer {tracer_id}: Job failed due to {error}")
        job_state = BaseJobState.FAILED
        try:
            shutil.rmtree(work_dir)
            print(f"successfully deleted {work_dir} directory")
        except Exception as e:
            logger.warning("Could not delete tmp directory, exiting")

        #job.messages.append(f"CO_level: FAILED. Unable to scrape data. {e}")


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
        except FileNotFoundError:
            logger.error(f"File not found in MinIO: {relative_path}")
            return res
    #TODO section : Not yet implemented
    # Handle JSON downloads from Webcam/API-scraped
    elif "Webcam/API-scraped" in source_data.relative_path and os.path.basename(source_data.relative_path) == "data.json":
        twitter_coords_path = os.path.join(work_dir, "Webcam", file_name)
        try:
            scraped_data_repository.download_json(source_data, job_id, twitter_coords_path)
            res["Webcam"] = True
        except FileNotFoundError:
            logger.error(f"File not found in MinIO: {relative_path}")
            return res

    # Handle image downloads from Webcam/scraped
    elif "Webcam/scraped" in source_data.relative_path:
        image_save_path = os.path.join(work_dir, "Webcam", "scraped_images", file_name)
        
        try:
            # Assuming there's a method in scraped_data_repository to download images
            scraped_data_repository.download_image(source_data, job_id, image_save_path)
            logger.info(f"Downloaded image to {image_save_path}")
        except FileNotFoundError:
            logger.error(f"Image file not found in MinIO: {relative_path}")
            return res

    return res


#TODO: needs to be redefined for image augmentations
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
