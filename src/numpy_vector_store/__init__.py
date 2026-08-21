"""
NumPy Vector Store - A simple vector store implementation using NumPy.

This package provides a lightweight vector store implementation for storing
and searching vector embeddings using NumPy arrays.
"""

__version__ = "0.5.0"
__author__ = "Tim VanReenen"

from .vector_store import VectorHit, VectorStore

__all__ = ["VectorHit", "VectorStore"]
