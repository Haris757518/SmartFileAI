from database import DatabaseManager

db = DatabaseManager()
cursor = db.conn.cursor()

print("Clearing image-related tables...")

cursor.execute("DELETE FROM image_embeddings;")
cursor.execute("DELETE FROM face_embeddings;")
cursor.execute("DELETE FROM person_embeddings;")

db.conn.commit()

print("Image tables cleared successfully.")