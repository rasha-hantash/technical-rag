"""Entry point for the PDF RAG server."""

import argparse
import os

import uvicorn

from pdf_llm_server.server import app


def main():
    parser = argparse.ArgumentParser(description="PDF RAG server")
    parser.add_argument(
        "--pdf-parser",
        choices=["pymupdf", "reducto"],
        default=None,
        help="PDF parser backend (default: pymupdf). Overrides PDF_PARSER env var.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    if args.pdf_parser:
        os.environ["PDF_PARSER"] = args.pdf_parser

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
