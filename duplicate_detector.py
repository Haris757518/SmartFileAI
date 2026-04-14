import os
import hashlib
from collections import defaultdict
from PIL import Image

try:
	import imagehash
except Exception:
	imagehash = None


# ---------------- CONFIG ----------------
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
HASH_BUFFER_SIZE = 65536
PHASH_THRESHOLD = 8   # lower = stricter similarity
# ----------------------------------------


def get_image_files(folder_path):
	image_files = []
	for root, _, files in os.walk(folder_path):
		for file in files:
			if file.lower().endswith(IMAGE_EXTENSIONS):
				image_files.append(os.path.join(root, file))
	return sorted(image_files)


# -------- EXACT DUPLICATE (SHA-256) --------
def compute_file_hash(file_path):
	sha256 = hashlib.sha256()
	try:
		with open(file_path, "rb") as f:
			while chunk := f.read(HASH_BUFFER_SIZE):
				sha256.update(chunk)
		return sha256.hexdigest()
	except OSError:
		return None


def find_exact_duplicates(image_files):
	hash_map = defaultdict(list)
	for path in image_files:
		file_hash = compute_file_hash(path)
		if file_hash:
			hash_map[file_hash].append(path)
	return hash_map


# -------- SIMILAR DUPLICATE (pHash) --------
def compute_phash(image_path):
	if imagehash is None:
		raise RuntimeError(
			"The 'imagehash' package is required for similar-image detection. "
			"Install it with: pip install ImageHash"
		)

	try:
		with Image.open(image_path) as img:
			return imagehash.phash(img)
	except OSError:
		return None


def find_similar_duplicates(image_files, phash_threshold=PHASH_THRESHOLD):
	representatives = []
	similar_groups = defaultdict(list)

	for path in image_files:
		ph = compute_phash(path)
		if ph is None:
			continue

		found = False
		for existing_ph in representatives:
			if ph - existing_ph <= phash_threshold:
				similar_groups[str(existing_ph)].append(path)
				found = True
				break

		if not found:
			representatives.append(ph)
			similar_groups[str(ph)].append(path)

	return similar_groups


def _to_duplicate_groups(group_map):
	groups = []
	for files in group_map.values():
		unique_files = sorted(set(files))
		if len(unique_files) > 1:
			groups.append(unique_files)

	groups.sort(key=lambda g: (-len(g), g[0].lower()))
	return groups


def scan_folder_duplicates(folder_path, mode="hybrid", phash_threshold=PHASH_THRESHOLD):
	if not os.path.isdir(folder_path):
		raise ValueError("Invalid folder path")

	if mode not in {"exact", "similar", "hybrid"}:
		raise ValueError("Mode must be 'exact', 'similar', or 'hybrid'")

	image_files = get_image_files(folder_path)
	result = {
		"folder_path": folder_path,
		"mode": mode,
		"total_images": len(image_files),
		"exact_groups": [],
		"similar_groups": [],
	}

	if mode in {"exact", "hybrid"}:
		exact_map = find_exact_duplicates(image_files)
		result["exact_groups"] = _to_duplicate_groups(exact_map)

	if mode in {"similar", "hybrid"}:
		similar_map = find_similar_duplicates(image_files, phash_threshold=phash_threshold)
		result["similar_groups"] = _to_duplicate_groups(similar_map)

	return result


# -------- DISPLAY --------
def display(title, groups):
	print(f"\n===== {title} =====\n")
	count = 0
	for files in groups:
		if len(files) > 1:
			count += 1
			print(f"Group {count}:")
			for f in files:
				print(f"  - {f}")
			print()
	if count == 0:
		print("No matches found.")


# -------- MAIN --------
if __name__ == "__main__":
	folder = input("Enter folder path: ").strip()

	if not os.path.exists(folder):
		print("Invalid folder path.")
		raise SystemExit(1)

	print("\nChoose Mode:")
	print("1. Exact duplicates only")
	print("2. Similar images only")
	print("3. Hybrid (Exact + Similar)")
	choice = input("Enter choice (1/2/3): ").strip()

	mode_map = {
		"1": "exact",
		"2": "similar",
		"3": "hybrid",
	}

	if choice not in mode_map:
		print("Invalid choice.")
		raise SystemExit(1)

	try:
		result = scan_folder_duplicates(folder, mode=mode_map[choice])
	except Exception as e:
		print(f"Error: {e}")
		raise SystemExit(1)

	print(f"\nScanning {result['total_images']} images...\n")

	if mode_map[choice] in {"exact", "hybrid"}:
		display("EXACT DUPLICATES", result["exact_groups"])

	if mode_map[choice] in {"similar", "hybrid"}:
		display("SIMILAR IMAGES", result["similar_groups"])
