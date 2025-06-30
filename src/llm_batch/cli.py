import json
import math

import fitz
import openai
import yaml
import logging
import logging.config
import polars as pl

from pathlib import Path, PosixPath
from dotenv import load_dotenv, find_dotenv
from rich.progress import track
from rich.console import Console
from datetime import datetime
from importlib import resources
from typing import List, Dict
from cyclopts import App, Parameter
from typing_extensions import Annotated

from llm_batch import data
from llm_batch import __version__


# ---------------------------------------------------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------------------------------------------------

load_dotenv()


help_msg = """
Commands to execute LLM batcb jobs.
"""
app = App(help=help_msg, version=__version__)


console = Console(style="white on black")


CONFIG = {}
with resources.path(data, "config.yml") as path:
    CONFIG = yaml.load(open(path), Loader=yaml.FullLoader)

# setup logging
logging.config.dictConfig(CONFIG["logging"])
logger = logging.getLogger(__name__)


# add sub-apps
# batch_app = App(help="Help string for the asynchronous batch application.", version=__version__)
# app.command(batch_app, name="batch")
# utils_app = App(help="Utility commands for supporting batch jobs", version=__version__)
# app.command(utils_app, name="utils")


# ---------------------------------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------------------------------
@app.command()
def config() -> None:
    "Display configuration parameters"
    console.print(CONFIG)

# ---------------------------------------------------------------------------------------------------------------------
@app.command()
def make(
    prompt_template_file: Annotated[Path, Parameter(help="Prompt template file")] = None,
    data_file: Annotated[Path, Parameter(help="Data file")] = None,
    id_col: Annotated[str, Parameter(help="Column name for the id")] = "id",
    out: Annotated[Path, Parameter(help="Path to output file")] = Path("."),
    batch_name: Annotated[str, Parameter("--batch", help="Batch name")] = "batch",
) -> None:
    """
    Make a batch file for uploading to OpenAI
    """
    pass
    # # Read the prompt template file
    # prompt_template = prompt_template_file.read_text()

    # # Read the data file
    # df = pl.DataFrame()
    # print(f"Data file: {data_file}, {data_file.suffix}")
    # if data_file.suffix == ".csv":
    #     df = pl.read_csv(data_file)
    # elif data_file.suffix == ".xlsx":
    #     df = pl.read_excel(data_file)
    # if id_col not in df.columns:
    #     df = df.with_row_index(name=id_col)

    # for col in df.columns:
    #     if col.startswith("file"):
    #         df = df.with_columns(pl.col(col).map_elements(lambda x: Path(x).read_text()))

    # print(df.head())

    # # Create the output file
    # if not out.exists():
    #     out.mkdir(parents=True)
    # out_file = out / f"{batch_name}-requests.jsonl"
    # out_file.write_text("")

    # # Loop through the data to create a jsonl batch file
    # requests = []
    # data: List[Dict] = df.to_dicts()
    # for index in track(range(len(data)), description="Processing..."):
    #     try:
    #         body = chevron.render(prompt_template, data[index])
    #         # body = json.loads(body)
    #         print(body)
    #         request = {
    #             "custom_id": f"id_{data[index][id_col]}",
    #             "method": "POST",
    #             "url": "/v1/chat/completions",
    #             "body": body,
    #         }
    #         requests.append(request)
    #     except Exception as e:
    #         console.print(f"\nError processing row {index}: {e}")
    #         return

    # out_file.write_text("\n".join([json.dumps(r) for r in requests]))
    # console.print(f"Batch file created: {out_file}")


# ---------------------------------------------------------------------------------------------------------------------
@app.command()
def send(
    batch_file: Annotated[Path, Parameter(help="Batch file")] = None,
):
    """
    Upload a batch file to OpenAI
    """
    client = openai.OpenAI()
    batch_input_file = client.files.create(file=open(batch_file, "rb"), purpose="batch")
    console.print(f"Uploaded batch file: {batch_file}")
    console.print(f"[orange1]{batch_input_file}")
    logger.info(f"Uploaded batch file: {batch_file}")
    logger.info(f"{batch_input_file}")


# ---------------------------------------------------------------------------------------------------------------------
@app.command()
def start(
    batch_file_id: Annotated[str, Parameter(help="Batch file ID")] = None,
    description: Annotated[str, Parameter("--desc", help="Description of the batch job")] = "batch job",
):
    """
    Start a batch job on OpenAI
    """
    client = openai.OpenAI()
    batch_create_response = client.batches.create(
        input_file_id=batch_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )
    logger.info(batch_create_response)
    console.print(batch_create_response)


# ---------------------------------------------------------------------------------------------------------------------
@app.command()
def fetch(
    batch_id: Annotated[str, Parameter(help="Batch ID")] = None,
    out: Annotated[Path, Parameter("--out", "-o", help="Path to output file")] = Path("."),
    batch_name: Annotated[str, Parameter("--batch", help="Batch name")] = "batch",
):
    """
    Download batch results to a file if the batch job is completed, else job status is displayed.
    """
    client = openai.OpenAI()
    batch_retrieve_response = client.batches.retrieve(batch_id)
    logger.info(batch_retrieve_response)
    console.print(batch_retrieve_response)
    if batch_retrieve_response.status == "completed":
        file_response = client.files.content(batch_retrieve_response.output_file_id)
        out.mkdir(parents=True, exist_ok=True)
        out_file = out / f"{batch_name}-responses.jsonl"
        out_file.write_text(file_response.text)
        logger.info(f"writing json output to {out_file}")
        console.print(f"[orange1]writing json output to {out_file}")


# ---------------------------------------------------------------------------------------------------------------------
@app.command()
def list(
    limit: Annotated[int, Parameter("--limit", "-l", help="Limit the number of batches to list")] = 100,
):
    """
    List all OpenAI batches for your account
    """
    client = openai.OpenAI()
    batches = client.batches.list(limit=limit)
    batches = sorted(batches, key=lambda x: x.created_at)
    for b in batches:
        console.print(b.id, b.status, datetime.fromtimestamp(b.created_at))


# ---------------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    app()
