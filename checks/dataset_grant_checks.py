import logging
from nameparser import HumanName

logger = logging.getLogger(__name__)

def get_dataset_grant_pi_matches(dataset_pis, grant_pis):
    """Checks whether the dataset and grant have matching PIs

    Args:
        dataset_pis (list): The dataset's list of PIs
        grant_pis (list): The grant's list of PIs
    """

    # Skip matching if dataset or grant has no PIs
    if None in [dataset_pis, grant_pis]:
        return None

    dataset_name_tuples = [(name.first, name.last) for name in map(HumanName, dataset_pis)]
    grant_name_tuples = [(name.first, name.last) for name in map(HumanName, grant_pis)]
    common = set(dataset_name_tuples) & set(grant_name_tuples)

    if common:
        logger.info(f"***************************************************************************************************************")
        logger.info(f"PI match found between dataset and grant")
        logger.info(f"dataset_pi:                      '{dataset_pis}'")
        logger.info(f"grant_pi_list:                 '{grant_pis}'")

    return ', '.join(f'{first} {last}' for first, last in common)
