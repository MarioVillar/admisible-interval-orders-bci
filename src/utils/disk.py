import os
import dotenv
import logging

import pandas as pd

dotenv.load_dotenv()


def get_disk_path() -> str:
    """
    Load the path to the disk where the profiles are stored.
    If no path is provided in the .env file, the default path is "localDB" in the current folder.

    If the directory does not exist, it is created.

    :return: str
    """
    disk_path = f"{os.getcwd()}/localDB"

    if os.getenv("DISK_PATH") is not None:
        disk_path = os.getenv("DISK_PATH")

    # Check if the directory exists
    check_create_folder(disk_path)

    return disk_path


def check_create_folder(folder_path: str) -> None:
    """
    Check if the folder exists. If not, create it.

    Parameters
    ----------
    folder_path : str
        The path to the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logging.warning(f"Directory {folder_path} did not exist. It has been automatically created.")


def check_create_file(filename: str) -> None:
    """
    Check if the file exists. If not, create it.

    Parameters
    ----------
    filename : str
        The path to the file.
    """
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w"):
            pass


def save_results_csv(
    results: pd.DataFrame,
    csv_path: str,
    overwrite: bool = False,
) -> None:
    """
    Save the results to disk. Two options are available:
    1. Appending to the csv file and overwrites the rows existing for the same
        "samples", "subject", "session", "channels", "n_sessions", "dataset", "pipeline" and "subj".
    2. Overwriting the csv file.

    Parameters
    ----------
    results : pd.DataFrame
        The results to be saved.
    csv_path : str
        The path to the csv file.
    """
    check_create_file(csv_path)

    if overwrite:
        results.to_csv(csv_path, index=False)
    else:
        # Probar a añadir datos
        try:
            df = pd.read_csv(csv_path)

            df = pd.concat([df, results], ignore_index=True)

            col_drop_dup = ["samples", "subject", "session", "channels", "n_sessions", "dataset", "pipeline"]

            if set(col_drop_dup).issubset(df.columns):
                df.drop_duplicates(
                    subset=col_drop_dup,
                    inplace=True,
                    keep="last",
                )

            df.to_csv(csv_path, index=False)
        # Si el csv está completamente vacío (sin cabeceras ni nada) entonces se escribe directamente
        except pd.errors.EmptyDataError:
            results.to_csv(csv_path, index=False)
