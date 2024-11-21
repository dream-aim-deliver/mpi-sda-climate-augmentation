from typing import Any, NamedTuple, Optional, Tuple
import requests
from datetime import datetime, timedelta
import re



def date_range(start_date, end_date):
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    date_list = [((start_date + timedelta(days=x)).strftime("%Y-%m-%d"),
                  (start_date + timedelta(days=x+1)).strftime("%Y-%m-%d"))
                 for x in range((end_date-start_date).days + 1)]
    return date_list


def load_evalscript(source: str) -> str:
    if source.startswith("http://") or source.startswith("https://"):
        # Load from URL
        response = requests.get(source)
        response.raise_for_status()  # Raise an exception if the request failed
        return response.text
    else:
        # Load from file
        with open(source, "r") as file:
            return file.read()


def sanitize_filename(filename):   #helper function
    return re.sub(r'[^\w./]', '_', filename)


class KernelPlancksterRelativePath(NamedTuple):
    case_study_name: str
    tracer_id: str
    job_id: str
    timestamp: str
    dataset: str
    evalscript_name: str
    image_hash: str
    file_extension: str

def generate_relative_path(case_study_name, tracer_id, job_id, timestamp, dataset, evalscript_name, image_hash, file_extension):
    return f"{case_study_name}/{tracer_id}/{job_id}/{timestamp}/sentinel/{dataset}_{evalscript_name}_{image_hash}.{file_extension}"

def parse_relative_path(relative_path) -> KernelPlancksterRelativePath:
    parts = relative_path.split("/")
    case_study_name = parts[0]
    tracer_id = parts[1]
    job_id = parts[2]
    timestamp = parts[3]
    dataset, evalscript_name, image_hash_extension = parts[5].split("_")
    image_hash, file_extension = image_hash_extension.split(".")
    return KernelPlancksterRelativePath(case_study_name, tracer_id, job_id, timestamp, dataset, evalscript_name, image_hash, file_extension)


def get_webcam_info_from_name(webcam_name: str) -> dict[str, str]:
    
    webcam_split = webcam_name.split("..")


    webcam_dict = {
        "location": webcam_split[0],
        "country": webcam_split[1],
        "latitude": webcam_split[2],
        "longitude": webcam_split[3]
    }

    return webcam_dict
