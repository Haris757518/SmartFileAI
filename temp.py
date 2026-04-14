import os
from image_indexer import ImageIndexer


def main():
    folder = input("Enter folder to reindex images: ").strip()

    if not os.path.exists(folder):
        print("Invalid folder path.")
        return

    print("\nStarting image reindex...\n")

    indexer = ImageIndexer()
    indexer.index_folder(folder)

    print("\nImage reindex completed successfully.\n")


if __name__ == "__main__":
    main()