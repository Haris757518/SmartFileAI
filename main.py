from app_core import AppCore
from file_watcher import start_watcher
import os


def print_text_results(results):
    if not results:
        print("\nNo matching documents found.\n")
        return

    for i, r in enumerate(results):
        print(f"\n[{i}] {r['file_name']}")
        print(f"  Score:   {r['final_score']:.4f}")
        print(f"  Snippet: {r['snippet'][:250]}...")


def print_image_results(results):
    if not results:
        print("\nNo matching images found.\n")
        return

    for i, r in enumerate(results):
        print(f"\n[{i}] {r['file_name']}")
        print(f"  Score: {r['final_score']:.4f}")
        print(f"  Path:  {r['file_path']}")


def main():

    folder_path = input("Enter folder path to watch: ").strip()

    if not os.path.exists(folder_path):
        print("Invalid folder path.")
        return

    print("\nInitializing SmartFileAI...\n")

    # Initialize search system
    app = AppCore()

    # Start real-time watcher
    observer = start_watcher(folder_path, app)

    print("\nType:")
    print("1 → Text Search")
    print("2 → Text → Image Search")
    print("3 → Image → Image Similarity")
    print("4 → Face Search")
    print("exit → Close program\n")

    try:
        while True:
            mode = input("Select mode (1/2/3/4): ").strip()
               
            if mode.lower() == "exit":
                print("Exiting SmartFileAI...")
                break

            # -------------------------
            # TEXT SEARCH
            # -------------------------
            if mode == "1":
                query = input("Enter search query: ").strip()
                results = app.search_text(query)
                print_text_results(results)

            # -------------------------
            # TEXT → IMAGE SEARCH
            # -------------------------
            elif mode == "2":
                query = input("Enter image search query: ").strip()
                results = app.search_images_by_text(query)
                print_image_results(results)

            # -------------------------
            # IMAGE → IMAGE SEARCH
            # -------------------------
            elif mode == "3":
                image_path = input("Enter full image path: ").strip()

                if not os.path.exists(image_path):
                    print("Image path invalid.\n")
                    continue

                results = app.search_similar_images(image_path)
                print_image_results(results)
            elif mode == "4":
                image_path = input("Enter image path containing face: ").strip()

                if not os.path.exists(image_path):
                    print("Invalid path.\n")
                    continue

                results = app.search_by_face(image_path)

                if not results:
                    print("\nNo matching faces found.\n")
                else:
                    for i, r in enumerate(results):
                        print(f"\n[{i}] {r['file_path']}")
                        print(f"  Score: {r['final_score']:.4f}")
            else:
                print("Invalid option.\n")

    except KeyboardInterrupt:
        print("\nStopping SmartFileAI...")

    finally:
        observer.stop()
        observer.join()
        print("Watcher stopped. Goodbye.")


if __name__ == "__main__":
    main()