from pathlib import Path
import base64
import shutil


def encode_filename(filename):
    """Obscure the filename by encoding it with base64."""
    encoded_bytes = base64.urlsafe_b64encode(filename.encode("utf-8"))
    encoded_str = encoded_bytes.decode("utf-8")
    return encoded_str


def decode_filename(encoded_str):
    """Decode the base64-encoded filename."""
    decoded_bytes = base64.urlsafe_b64decode(encoded_str.encode("utf-8"))
    decoded_str = decoded_bytes.decode("utf-8")
    return decoded_str


def obscure_files(src_folder, dest_folder):
    """Traverse the source folder, obscure the filenames, and copy to the destination folder."""
    src_path = Path(src_folder)
    dest_path = Path(dest_folder)

    # Ensure destination folder exists
    dest_path.mkdir(parents=True, exist_ok=True)

    for path in src_path.rglob("*"):  # Recursively walk through the directory
        if path.is_file():
            # Obscure filenames but keep extensions
            filename = path.stem  # Filename without extension
            if "melia" in str(path) or any(
                [str(year) in str(path) for year in range(2018, 2025)]
            ):
                ext = path.suffix  # File extension

                obscured_name = encode_filename(filename) + ext

                # Reconstruct relative path in the destination directory
                relative_path = path.relative_to(src_path)
                new_file_path = dest_path / relative_path.parent / obscured_name

                # Create any required parent directories
                new_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy the file to the new destination
                shutil.copy2(path, new_file_path)


# Usage:
src_folder = "/Users/vigji/Desktop/sando-data/audio"
dest_folder = "/Users/vigji/Desktop/sando-data/new-audio"

obscure_files(src_folder, dest_folder)
