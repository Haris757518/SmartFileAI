import sqlite3
import os
import numpy as np


class DatabaseManager:
    def __init__(self):
        os.makedirs("data", exist_ok=True)
        self.db_path = os.path.join("data", "index.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

        # Improve performance + stability
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")

        self.create_tables()

    # =========================================================
    # CREATE TABLES
    # =========================================================
    def create_tables(self):
        cursor = self.conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS text_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT,
            file_name TEXT,
            file_hash TEXT,
            last_modified REAL,
            chunk_text TEXT,
            embedding BLOB
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS image_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT,
            file_name TEXT,
            file_hash TEXT,
            last_modified REAL,
            embedding BLOB
        )
        """)
        # Face embeddings table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT,
            embedding BLOB,
            gender TEXT
        )
        """)

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_face_path ON face_embeddings(file_path)"
        )
        # Person crop embeddings table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS person_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT,
            embedding BLOB
        )
        """)


        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_person_path ON person_embeddings(file_path)"
        )
        # Indexes for faster lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_text_path ON text_chunks(file_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_text_hash ON text_chunks(file_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_path ON image_embeddings(file_path)")

        self.conn.commit()

    # =========================================================
    # TEXT INSERT
    # =========================================================
    def insert_text_chunk(self, file_path, file_name, file_hash,
                      last_modified, chunk_text, embedding):

        cursor = self.conn.cursor()

        embedding_blob = embedding.astype(np.float32).tobytes()

        cursor.execute("""
        INSERT INTO text_chunks
        (file_path, file_name, file_hash, last_modified, chunk_text, embedding)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (file_path, file_name, file_hash,
            last_modified, chunk_text, embedding_blob))

        self.conn.commit()

    # =========================================================
    # INSERT PERSON EMBEDDING
    # =========================================================
    def insert_person_embedding(self, file_path, embedding):
        cursor = self.conn.cursor()
        embedding_blob = embedding.astype(np.float32).tobytes()
        cursor.execute("""
        INSERT INTO person_embeddings
        (file_path, embedding)
        VALUES (?, ?)
        """, (file_path, embedding_blob))
        self.conn.commit()

    # =========================================================
    # FETCH PERSON EMBEDDINGS
    # =========================================================
    def fetch_all_person_embeddings(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT file_path, embedding FROM person_embeddings")
        rows = cursor.fetchall()

        results = []
        for row in rows:
            embedding = np.frombuffer(row[1], dtype=np.float32)
            results.append((row[0], embedding))

        return results   
    # =========================================================
    # INSERT PERSON EMBEDDING
    # =========================================================
    def insert_person_embedding(self, file_path, embedding):
        cursor = self.conn.cursor()
        embedding_blob = embedding.astype(np.float32).tobytes()
        cursor.execute("""
        INSERT INTO person_embeddings
        (file_path, embedding)
        VALUES (?, ?)
        """, (file_path, embedding_blob))
        self.conn.commit()

    # =========================================================
    # FETCH PERSON EMBEDDINGS
    # =========================================================
    def fetch_all_person_embeddings(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT file_path, embedding FROM person_embeddings")
        rows = cursor.fetchall()

        results = []
        for row in rows:
            embedding = np.frombuffer(row[1], dtype=np.float32)
            results.append((row[0], embedding))

        return results
    # =========================================================
    # IMAGE INSERT
    # =========================================================
    def insert_image(self, file_path, file_name,
                     file_hash, last_modified, embedding):

        cursor = self.conn.cursor()

        embedding_blob = embedding.astype(np.float32).tobytes()

        cursor.execute("""
        INSERT INTO image_embeddings
        (file_path, file_name, file_hash, last_modified, embedding)
        VALUES (?, ?, ?, ?, ?)
        """, (file_path, file_name,
              file_hash, last_modified, embedding_blob))

        self.conn.commit()

    # =========================================================
    # FETCH TEXT
    # =========================================================
    def fetch_all_text_chunks(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM text_chunks")
        rows = cursor.fetchall()

        results = []
        for row in rows:
            embedding = np.frombuffer(row[6], dtype=np.float32)
            results.append((
                row[1],  # file_path
                row[2],  # file_name
                row[3],  # file_hash
                row[4],  # last_modified
                row[5],  # chunk_text
                embedding
            ))

        return results

    # =========================================================
    # FETCH IMAGES
    # =========================================================
    def fetch_all_images(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM image_embeddings")
        rows = cursor.fetchall()

        results = []
        for row in rows:
            embedding = np.frombuffer(row[5], dtype=np.float32)
            results.append((
                row[1],  # file_path
                row[2],  # file_name
                row[3],  # file_hash
                row[4],  # last_modified
                embedding
            ))

        return results

    # =========================================================
    # DELETE IMAGE / FACE / PERSON ROWS BY FILE
    # =========================================================
    def delete_image(self, file_path):
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM image_embeddings WHERE file_path = ?",
            (file_path,)
        )
        self.conn.commit()

    def delete_face_embeddings(self, file_path):
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM face_embeddings WHERE file_path = ?",
            (file_path,)
        )
        self.conn.commit()

    def delete_person_embeddings(self, file_path):
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM person_embeddings WHERE file_path = ?",
            (file_path,)
        )
        self.conn.commit()

    # =========================================================
    # GET FILE HASH
    # =========================================================
    def get_file_hash(self, file_path):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT file_hash FROM text_chunks WHERE file_path = ? LIMIT 1",
            (file_path,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    # =========================================================
    # GET ALL INDEXED FILES
    # =========================================================
    def get_all_indexed_files(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT file_path FROM text_chunks")
        rows = cursor.fetchall()
        return {row[0] for row in rows}

    # =========================================================
    # DELETE TEXT CHUNKS
    # =========================================================
    def delete_file_chunks(self, file_path):
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM text_chunks WHERE file_path = ?",
            (file_path,)
        )
        self.conn.commit()

    # =========================================================
    # CLEAR TEXT INDEX (SAFE RESET)
    # =========================================================
    def clear_text_index(self):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM text_chunks")
        self.conn.commit()
        print("Text index cleared.")
    # =========================================================
    # INSERT FACE
    # =========================================================
    def insert_face_embedding(self, file_path, embedding, gender):
        cursor = self.conn.cursor()
        embedding_blob = embedding.astype(np.float32).tobytes()
        cursor.execute("""
        INSERT INTO face_embeddings
        (file_path, embedding, gender)
        VALUES (?, ?, ?)
        """, (file_path, embedding_blob, gender))
        self.conn.commit()

    # =========================================================
    # FETCH FACES
    # =========================================================
    def fetch_all_faces(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT file_path, embedding, gender FROM face_embeddings")
        rows = cursor.fetchall()

        results = []
        for row in rows:
            embedding = np.frombuffer(row[1], dtype=np.float32)
            gender = row[2]
            results.append((row[0], embedding, gender))

        return results