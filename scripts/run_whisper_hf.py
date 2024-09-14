from pathlib import Path

# import whisper
from tqdm import tqdm
from pprint import pprint
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time

# CHANGE THIS TO YOUR DATA PATH:
data_path = Path("/Users/vigji/Desktop/sando-data/audio")
data_output = data_path / "whispered"


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


def process_all_files(source_files, data_path, output_path=None):
    for source_file in source_files:
        if output_path is None:
            output_path = source_file.parent / get_output_filename(
                source_file, data_path
            )


if __name__ == "__main__":
    # if running from terminal there is an argument, that is the data path:
    import sys

    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])

    torch_device = "cpu"
    if len(sys.argv) > 2:
        torch_device = sys.argv[2]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs={"model_max_length": 81810},
    )

    # model = whisper.load_model("large-v3", device=torch_device)
    # model.transcribe = torch.compile(model.transcribe, backend="inductor")

    # check if device is available:
    if torch_device == "mps":
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print(
                    "MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                )
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )
                raise RuntimeError("MPS not available")
    elif torch_device == "cuda":
        assert torch.cuda.is_available(), f"Device {torch_device} not available"
    elif torch_device == "cpu":
        pass
    else:
        raise ValueError(f"Invalid device {torch_device}")

    # find all files called *omelia*.mp3 in data_path subfolders until 2017:
    all_omelie = list(data_path.rglob("*melia*.mp3"))

    # add all mp3 files from 2018 onwards:
    all_omelie += list(
        [
            f
            for f in data_path.rglob("*.mp3")
            if any([str(year) in str(f) for year in range(2018, 2025)])
        ]
    )

    all_omelie = sorted(all_omelie)[::-1]
    print("Found files: ", len(all_omelie))

    # Filter beforehand to have truetful progress bar:
    to_process = [
        str(f)
        for f in all_omelie
        if not (data_output / get_output_filename(f, data_path)).exists()
    ]

    start_time = time.time()
    print(start_time)
    result = pipe(
        to_process[0], return_timestamps=True, generate_kwargs={"language": "italian"}
    )
    # batch_size=5)

    # to be sure, pickle and save the result:
    end_time = time.time()
    print(end_time)
    print("Time elapsed: ", end_time)

    print(result)
    print(result.keys())
    import pickle

    with open(data_path / "cached_result.pkl", "wb") as f:
        pickle.dump(result, f)
    print(result["segments"])

    # print(result["timestamps"])

    # for input_file in tqdm(to_process):
    #     output_filename = data_output / get_output_filename(input_file, data_path)

    #     # Call whisper model:
    #     with torch.device(torch_device):

    #         result = model.transcribe(
    #             str(input_file),
    #             initial_prompt="Un'omelia registrata durante una messa.",
    #         )

    #     # Concatenate fragments and save results:
    #     full_text = [t["text"] for t in result["segments"]]

    #     with open(output_filename, "w") as f:
    #         f.write("\n".join(full_text))
