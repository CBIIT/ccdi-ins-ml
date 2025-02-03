import logging
logger = logging.getLogger(__name__)

def get_dataset_program_funding_matches(dataset_description, dataset_funding_source_list, program_awards_list, program_nofo_list):
    """Checks whether the dataset and program have matching funding

    Args:
        dataset_description (str): The dataset's description
        dataset_funding_source_list (list): The dataset's funding sources
        program_awards_list (list): The program's awards
        program_nofo_list (list): The program's notices of funding opportunities
    """

    matches = {}

    for award in program_awards_list:
        if award in dataset_description:
            if "awards_to_desc" not in matches:
                matches["awards_to_desc"] = {
                    "program_awards": [],
                    "dataset_description": dataset_description
                }

            matches["awards_to_desc"]["program_awards"].append(award)

        for funding_source in dataset_funding_source_list:
            if award not in funding_source:
                continue

            matches["awards_to_fs"] = []

            matches["awards_to_fs"].append({
                "program_award": award,
                "dataset_funding_source": funding_source
            })

    for nofo in program_nofo_list:
        if nofo in dataset_description:
            if "nofos_to_desc" not in matches:
                matches["nofos_to_desc"] = {
                    "nofos": [],
                    "dataset_description": dataset_description
                }

            matches["nofos_to_desc"]["nofos"].append(nofo)

        for funding_source in dataset_funding_source_list:
            if nofo not in funding_source:
                continue

            matches["nofos_to_fs"] = []

            matches["nofos_to_fs"].append({
                "program_nofo": nofo,
                "dataset_funding_source": funding_source
            })

    if matches:
        logger.info(f"*******************************************************************************************************")
        logger.info(f"Funding source match found between dataset and  program")
        logger.info(f"dataset_description:            '{dataset_description}'")
        logger.info(f"dataset_funding_source_list:    '{dataset_funding_source_list}'")
        logger.info(f"dataset_funding_source_list:    '{dataset_funding_source_list}'")
        logger.info(f"nofo:                           '{program_nofo_list}'")
        logger.info(f"award:                          '{program_awards_list}'")

    return matches

def get_dataset_program_name_matches(dataset_description, dataset_title, program_acronym, program_name):
    """Checks whether the dataset mentions the program's acronym or name

    Args:
        dataset_description (str): The dataset's description
        dataset_title (str): The dataset's title
        program_acronym (str): The program's acronym
        program_name (str): The program's name
    """

    matches = {}

    if program_acronym in dataset_description:
        matches["acr_to_desc"] = {
            "program_acronym": program_acronym,
            "dataset_description": dataset_description
        }

    if program_acronym in dataset_title:
        matches["acr_to_title"] = {
            "program_acronym": program_acronym,
            "dataset_title": dataset_title
        }

    if program_name in dataset_description:
        matches["name_to_desc"] = {
            "program_name": program_name,
            "dataset_description": dataset_description
        }

    if program_name in dataset_title:
        matches["name_to_title"] = {
            "program_name": program_name,
            "dataset_title": dataset_title
        }

    if matches:
        logger.info(f"*******************************************************************************************************")
        logger.info(f"Name/acronym match found between  dataset and  program")
        logger.info(f"dataset_description:            '{dataset_description}'")
        logger.info(f"dataset_title:                  '{dataset_title}'")
        logger.info(f"program_name:                   '{program_name}'")
        logger.info(f"program_acronym:                '{program_acronym}'")

    return matches

def get_dataset_program_pi_matches(dataset_pi, program_pi_list):
    """Checks whether the dataset and program have matching PIs

    Args:
        dataset_pi (str): The dataset's list of PIs
        program_pi_list (list): The program's list of PIs
    """

    matches = [pi for pi in dataset_pi if pi in program_pi_list]

    if matches:
        logger.info(f"***************************************************************************************************************")
        logger.info(f"PI match found between dataset  and  program")
        logger.info(f"dataset_pi:                      '{dataset_pi}'")
        logger.info(f"program_pi_list:                 '{program_pi_list}'")

    return matches
