"""Rebuild the fake-mice SQLite and XLSX fixtures from fake-mice.csv.

    python tests/fixtures/make_fake_mice.py

fake-mice.csv is the master copy. The SQLite file has one table, ``mice``,
with the CSV's columns in snake_case and NULL for an empty identifier. The
workbook's ``mice`` sheet stores every cell as text; its ``numeric-id`` sheet
is the same animals with FAKE-001's RFID stored as a number, which an import
must refuse.
"""

from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent
CSV_PATH = HERE / "fake-mice.csv"
SQLITE_PATH = HERE / "fake-mice.sqlite"
XLSX_PATH = HERE / "fake-mice.xlsx"


def read_rows():
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


def column_name(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", header.casefold()).strip("_")


def build_sqlite(path: Path = SQLITE_PATH) -> Path:
    headers, rows = read_rows()
    columns = [column_name(header) for header in headers]
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as db:
        db.execute(f"CREATE TABLE mice ({', '.join(c + ' TEXT' for c in columns)})")
        db.executemany(
            f"INSERT INTO mice VALUES ({', '.join('?' for _ in columns)})",
            [[value or None for value in row] for row in rows],
        )
    db.close()
    return path


def build_xlsx(path: Path = XLSX_PATH) -> Path:
    import openpyxl

    headers, rows = read_rows()
    rfid = headers.index("RFID")
    workbook = openpyxl.Workbook()
    text = workbook.active
    text.title = "mice"
    numeric = workbook.create_sheet("numeric-id")
    for sheet in (text, numeric):
        sheet.append(headers)
        for row in rows:
            sheet.append([value or None for value in row])
        for cells in sheet.iter_rows(min_row=2):
            for cell in cells:
                cell.number_format = "@"  # Text, as the docs ask for RFIDs.
    numeric.cell(row=2, column=rfid + 1).value = int(rows[0][rfid])
    numeric.cell(row=2, column=rfid + 1).number_format = "0"
    workbook.save(path)
    return path


if __name__ == "__main__":
    print(build_sqlite())
    print(build_xlsx())
