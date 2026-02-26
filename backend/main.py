"""Entry point for the Technical RAG server."""

import argparse

import uvicorn

from technical_rag.server import app


def main():
    parser = argparse.ArgumentParser(description="Technical RAG server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
