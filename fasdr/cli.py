import argparse
import os
import sys
import logging
import rich
from rich.console import Console
from rich.logging import RichHandler

from fasdr import DocumentIndex

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("rich")
logger.info("Hello, World!")

def main():
    console = Console()
    parser = argparse.ArgumentParser(description="Fasdr: Fast Approximate String Distance Retrieval")
    parser.add_argument("path", help="Path to the index directory")
    args = parser.parse_args()
    logger.debug(args)
    index_path = args.path
    logger.debug(index_path)
    logger.debug(sys.argv)
    if not os.path.exists(index_path):
        console.print(f"Error: Path '[red]{index_path}[/red]' does not exist.", style="bold red")
        sys.exit(1)

    with console.status("Building index..."):
        index = DocumentIndex(index_path)

    console.print("Done!", style="bold green")
    console.print(f"Index generated at [green]{index_path}/.embeddings[/green]", style="bold green")

if __name__ == "__main__":
    main()
    # console = Console()
    # parser = argparse.ArgumentParser(description="Fasdr: Fast Approximate String Distance Retrieval")
    # parser.add_argument("path", help="Path to the index directory")
    # args = parser.parse_args()

    # index_path = args.path
    # if not os.path.exists(index_path):
    #     console.print(f"Error: Path '[red]{index_path}[/red]' does not exist.", style="bold red")
    #     sys.exit(1)

    # with console.status("Building index..."):
    #     index = DocumentIndex(index_path)

    # console.print("Done!", style="bold green")
    # console.print(f"Index generated at [green]{index_path}/.embeddings[/green]", style="bold green")
