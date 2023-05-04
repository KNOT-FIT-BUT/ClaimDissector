# Written by Martin Fajcik <martin.fajcik@vut.cz>
#
"""
author: Martin Fajcik, drqa's authors

"""
import sqlite3


class PassageDB:
    """Sqlite backed document storage.

    Borrowed from drqa's code,
    Warning: the contents of this class are not secured against SQL injection attack
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

    def get_all(self, column_name, column_value, table, columns, fetch_all=False):
        """Fetch the raw text of the doc for 'doc_id'."""
        if type(columns) == list:
            columns = ", ".join(columns)

        cursor = self.connection.cursor()
        cursor.execute(
            f"SELECT {columns} FROM {table} WHERE {column_name} in ({','.join(['?'] * len(column_value))})",
            column_value
        )
        if fetch_all:
            result = cursor.fetchall()
        else:
            result = cursor.fetchone()
        cursor.close()
        if result is None:
            raise ValueError(f"Column named {column_name} with value:" + str(column_value) + " is not in the database!")
        return result

    def get_size(self, table_name):
        cursor = self.connection.cursor()
        cursor.execute(
            f"SELECT COUNT(*) FROM {table_name}"

        )
        result = cursor.fetchone()
        cursor.close()
        return result[0]

    def get(self, column_name, column_value, table, columns, fetch_all=False):
        """Fetch the raw text of the doc for 'doc_id'."""
        if type(columns) == list:
            columns = ", ".join(columns)

        cursor = self.connection.cursor()
        cursor.execute(
            f"SELECT {columns} FROM {table} WHERE {column_name} = ?",
            (column_value,)
        )
        if fetch_all:
            result = cursor.fetchall()
        else:
            result = cursor.fetchone()
        cursor.close()
        if result is None:
            raise ValueError(f"Column named {column_name} with value:" + str(column_value) + " is not in the database!")
        return result

    def get_doc_text(self, doc_id, table="paragraphs", columns="raw_paragraph_context"):
        return self.get(column_name='id', column_value=doc_id, table=table, columns=columns)

    def get_doc_ids(self, table="paragraphs"):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT id FROM {table}")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_titles(self, table="documents"):
        """Fetch all titles of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT document_title FROM {table}")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return set(results)
