import logging
import sys
from app.augment import augment
from app.sdk.scraped_data_repository import ScrapedDataRepository
from app.setup import setup



def main(
    case_study_name: str,
    job_id: int,
    tracer_id: str,
    work_dir: str,
    kp_auth_token: str,
    kp_host: str,
    kp_port: int,
    kp_scheme: str,
    log_level: str = "WARNING",
) -> None:

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=log_level)

    try:
    
        if not all([case_study_name, job_id, tracer_id]): 
            logger.error(f"{job_id}: job_id, tracer_id must all be set.") 
            raise ValueError("job_id, tracer_id must all be set.")

        kernel_planckster, protocol, file_repository = setup(
            job_id=job_id,
            logger=logger,
            kp_auth_token=kp_auth_token,
            kp_host=kp_host,
            kp_port=kp_port,
            kp_scheme=kp_scheme,
        )

        scraped_data_repository = ScrapedDataRepository(
            protocol=protocol,
            kernel_planckster=kernel_planckster,
            file_repository=file_repository,
        )

    except Exception as e:
        logger.error(f"Error setting up scraper: {e}")
        sys.exit(1)


    augment(
        case_study_name=case_study_name,
        job_id=job_id,
        tracer_id=tracer_id,
        scraped_data_repository=scraped_data_repository,
        log_level=log_level,
        work_dir = work_dir,
        protocol=protocol,
    )



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Download data from MinIO for climate based augmentations.")

    parser.add_argument(
        "--case-study-name",
        type=str,
        help="The case study name",
        required=True
    )

    parser.add_argument(
        "--job-id",
        type=int,
        help="The job id",
        required=True
    )

    parser.add_argument(
        "--tracer-id",
        type=str,
        help="The tracer id",
        required=True
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        help="The log level to use when running the pipeline. Possible values are DEBUG, INFO, WARNING, ERROR, CRITICAL. Set to WARNING by default.",
    )


    parser.add_argument(
        "--work_dir",
        type=str,
        default="./.tmp",
        help="work dir",
    )

    parser.add_argument(
        "--kp-auth-token",
        type=str,
        help="The Kernel Planckster auth token",
        required=True
    )

    parser.add_argument(
        "--kp-host",
        type=str,
        help="The Kernel Planckster host",
        required=True
    )

    parser.add_argument(
        "--kp-port",
        type=int,
        help="The Kernel Planckster port",
        required=True
    )

    parser.add_argument(
        "--kp-scheme",
        type=str,
        help="The Kernel Planckster scheme",
        required=True
    )
   

    args = parser.parse_args()

    main(
        case_study_name=args.case_study_name,
        job_id=args.job_id,
        tracer_id=args.tracer_id,
        log_level=args.log_level,
        work_dir = args.work_dir,
        kp_auth_token=args.kp_auth_token,
        kp_host=args.kp_host,
        kp_port=args.kp_port,
        kp_scheme=args.kp_scheme,
    )


