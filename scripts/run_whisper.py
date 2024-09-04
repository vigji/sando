from pathlib import Path
import whisper
from tqdm import tqdm
from pprint import pprint
import torch

# CHANGE THIS TO YOUR DATA PATH:
data_path = Path("/Users/vigji/Desktop/sando-data/audio")
torch_device = "mps"


def get_output_filename(source_filename, main_data_path):
    # different naminf pre and post 2018:
    # Get first subfolder after '/Users/vigji/Desktop/sando-data/audio' parent:
    year = int(source_filename.relative_to(main_data_path).parts[0])
    if year < 2018:
        word_list_candidates_for_date = source_filename.parent.name.split(" ")
        # keep only numbers in the date:
        datelist = [x for x in word_list_candidates_for_date if x.isdigit()]
    else:
        word_list_candidates_for_date = source_filename.stem.split(" ")
        # keep only numbers in the date:
        datelist = [x for x in word_list_candidates_for_date if x.isdigit()]
        # print(word_list_candidates_for_date)

    date = "-".join(datelist)
    return f"{date}_{source_filename.stem.replace(' ', '-')}_whispered.txt"


if __name__ == "__main__":
    # check if device is available:
    if torch_device == "mps":
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
                raise RuntimeError("MPS not available")
    else:
        assert torch.cuda.is_available(), f"Device {torch_device} not available"

    data_output = data_path / "whispered"

    # find all files called *omelia*.mp3 in data_path subfolders until 2017:
    all_omelie = list(data_path.rglob("*melia*.mp3"))

    # add all mp3 files from 2018 onwards:
    all_omelie += list([f for f in data_path.rglob("*.mp3") if any([str(year) in str(f) for year in range(2018, 2025)])])

    all_omelie = sorted(all_omelie)

    # pprint({f: get_output_filename(f, data_path) for f in all_omelie})

    model = whisper.load_model("large-v3", device=torch_device)

    # Filter beforehand to have truetful progress bar:
    to_process = [f for f in all_omelie if not (data_output / get_output_filename(f, data_path)).exists()]

    for input_file in tqdm(to_process):
        output_filename = data_output / get_output_filename(input_file, data_path)

        # Call whisper model:
        with torch.device(torch_device):
            result = model.transcribe(str(input_file), initial_prompt="Un'omelia registrata durante una messa.") 

        # Concatenate fragments and save results:
        full_text = [t["text"] for t in result["segments"]]

        with open(output_filename, "w") as f:
            f.write("\n".join(full_text)) 