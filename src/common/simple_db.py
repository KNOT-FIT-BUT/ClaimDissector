# Written by Martin Fajcik <martin.fajcik@vut.cz>
#
"""
author: Martin Fajcik, drqa's authors

"""
import sqlite3


class PassageDB:
    """Sqlite backed document storage.

    Borrowed from drqa's code
    """

    def __init__(self, db_path):
        self.path = db_path
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_text(self, doc_id, table="paragraphs", columns="raw_paragraph_context"):
        """Fetch the raw text of the doc for 'doc_id'."""
        if type(columns) == list:
            columns = ", ".join(columns)

        cursor = self.connection.cursor()
        cursor.execute(
            f"SELECT {columns} FROM {table} WHERE id = ?",
            (doc_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        if result is None:
            raise ValueError(f"ID {doc_id} not in the database!")
        return result

    def get_doc_ids(self, table="paragraphs"):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT id FROM {table}")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results
