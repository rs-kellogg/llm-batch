import json
import openai
from datetime import datetime
from pathlib import Path
from typing_extensions import Annotated
from cyclopts import App, Parameter
from llm_batch import (
    __version__,
    console,
    logger,
)

# ---------------------------------------------------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------------------------------------------------
openai_batch_app = App(help="OpenAI batching commands", version=__version__)


# ---------------------------------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------------------------------
@openai_batch_app.command()
def make(
    in_dir: Annotated[Path, Parameter(help="Path to input files")] = Path("."),
    out: Annotated[Path, Parameter(help="Path to output file")] = Path("."),
    batch_name: Annotated[str, Parameter("--batch", help="Batch name")] = "batch",
) -> None:
    """
    Make a batch file for uploading to OpenAI
    """
    json_files = [f for f in in_dir.glob("*.json")]
    if not json_files:
        console.print("[red]No JSON files found in the input directory.[/red]")
        return

    # Create the output file
    if not out.exists():
        out.mkdir(parents=True)
    out_file = out / f"{batch_name}-requests.jsonl"
    out_file.write_text("")

    # Loop through the JSON request files
    requests = []
    for f in json_files:
        try:
            console.print(f"[green]Found JSON file:[/green] {f}")
            request_body = json.loads(f.read_text())
            if "request" in request_body:
                request_body = request_body["request"]
            request = {
                "custom_id": f"id_{f.name}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request_body,
            }
            requests.append(request)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error decoding JSON in file {f}: {e}[/red]")
            continue
    out_file.write_text("\n".join([json.dumps(r) for r in requests]))
    console.print(f"Batch file created: {out_file}")


# ---------------------------------------------------------------------------------------------------------------------
@openai_batch_app.command()
def send(
    batch_file: Annotated[Path, Parameter(help="Batch file")] = None,  # type: ignore
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
@openai_batch_app.command()
def start(
    batch_file_id: Annotated[str, Parameter(help="Batch file ID")] = None,  # type: ignore
    description: Annotated[
        str, Parameter("--desc", help="Description of the batch job")
    ] = "batch job",
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
@openai_batch_app.command()
def fetch(
    batch_id: Annotated[str, Parameter(help="Batch ID")] = None,  # type: ignore
    out: Annotated[Path, Parameter("--out", help="Path to output file")] = Path("."),
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
        file_response = client.files.content(batch_retrieve_response.output_file_id)  # type: ignore
        out.mkdir(parents=True, exist_ok=True)
        out_file = out / f"{batch_name}-responses.jsonl"
        out_file.write_text(file_response.text)
        logger.info(f"writing json output to {out_file}")
        console.print(f"[orange1]writing json output to {out_file}")


# ---------------------------------------------------------------------------------------------------------------------
@openai_batch_app.command()
def check(
    limit: Annotated[
        int, Parameter("--limit", help="Limit the number of batches to list")
    ] = 100,
):
    """
    Display all OpenAI batches for your account
    """
    client = openai.OpenAI()
    batches = client.batches.list(limit=limit)
    batches = sorted(batches, key=lambda x: x.created_at)
    for b in batches:
        console.print(b.id, b.status, datetime.fromtimestamp(b.created_at))
