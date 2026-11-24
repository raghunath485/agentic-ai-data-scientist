"""
Multi-Source Quantum Data Ingestor.
Seamlessly loads data from CSV (auto-delimiter), Excel (.xlsx), Parquet, JSON, SQLite, and Web URLs.
"""

import os
import sqlite3
import pandas as pd
import requests
from typing import Dict, Any, List, Tuple, Optional


class MultiSourceIngestor:
    @staticmethod
    def load_from_file_or_buffer(
        file_buffer_or_path: Any,
        filename: str,
        sheet_name: Optional[str] = None,
        sqlite_table: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads dataset with automatic format and delimiter detection.
        """
        ext = os.path.splitext(filename)[1].lower()
        metadata = {"filename": filename, "format": ext}

        if ext == ".csv":
            try:
                df = pd.read_csv(file_buffer_or_path)
            except Exception:
                try:
                    if hasattr(file_buffer_or_path, "seek"):
                        file_buffer_or_path.seek(0)
                    df = pd.read_csv(file_buffer_or_path, sep=None, engine='python')
                except Exception:
                    if hasattr(file_buffer_or_path, "seek"):
                        file_buffer_or_path.seek(0)
                    df = pd.read_csv(file_buffer_or_path, encoding="latin1")

        elif ext in [".xlsx", ".xls"]:
            xl = pd.ExcelFile(file_buffer_or_path)
            metadata["available_sheets"] = xl.sheet_names
            target_sheet = sheet_name or xl.sheet_names[0]
            df = xl.parse(target_sheet)
            metadata["loaded_sheet"] = target_sheet

        elif ext == ".parquet":
            df = pd.read_parquet(file_buffer_or_path)

        elif ext in [".json", ".jsonl"]:
            try:
                df = pd.read_json(file_buffer_or_path)
            except Exception:
                if hasattr(file_buffer_or_path, "seek"):
                    file_buffer_or_path.seek(0)
                df = pd.read_json(file_buffer_or_path, lines=True)

        elif ext in [".sqlite", ".db"]:
            if isinstance(file_buffer_or_path, str):
                conn = sqlite3.connect(file_buffer_or_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [t[0] for t in cursor.fetchall()]
                metadata["available_tables"] = tables
                tbl = sqlite_table or tables[0]
                df = pd.read_sql_query(f"SELECT * FROM {tbl}", conn)
                conn.close()
            else:
                raise ValueError("SQLite ingestion requires a local file path.")

        else:
            # Fallback to standard CSV
            df = pd.read_csv(file_buffer_or_path)

        metadata["shape"] = df.shape
        return df, metadata

    @staticmethod
    def load_from_web_url(url: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Downloads and loads CSV or JSON directly from an HTTP URL."""
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        import io
        buf = io.BytesIO(resp.content)
        name = url.split("/")[-1].split("?")[0] or "dataset.csv"
        return MultiSourceIngestor.load_from_file_or_buffer(buf, name)
