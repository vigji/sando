import os
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from tqdm import tqdm
import time
import multiprocessing as mp
import pickle
import warnings
warnings.filterwarnings("ignore")

# CHANGE THIS TO YOUR DATA PATH:
data_path = Path("/Users/vigji/Desktop/sando-data/audio")
data_output = data_path / "whispered"


def get_output_filename(source_filename, main_data_path):
    year = int(source_filename.relative_to(main_data_path).parts[0])
    if year < 2018:
        word_list_candidates_for_date = source_filename.parent.name.split(" ")
        datelist = [x for x in word_list_candidates_for_date if x.isdigit()]
    else:
        word_list_candidates_for_date = source_filename.stem.split(" ")
        datelist = [x for x in word_list_candidates_for_date if x.isdigit()]

    date = "-".join(datelist)
    return f"{date}_{source_filename.stem.replace(' ', '-')}_whispered"


def process_file(file, pipe, data_path, progress_queue):
    txt_output_filename = data_output / (get_output_filename(file, data_path) + "text.txt")
    segment_output_filename = data_output / (
        get_output_filename(file, data_path) + "segments.pkl"
    )

    if not txt_output_filename.exists():
        result = pipe(
            str(file), return_timestamps=True, generate_kwargs={"language": "italian"}
        )
        with open(segment_output_filename, "wb") as f:
            pickle.dump(result["chunks"], f)
        with open(txt_output_filename, "w") as f:
            f.write(result["text"])

        # Notify the progress queue that one file is completed
        progress_queue.put(1)

        return txt_output_filename
    else:
        return None


def worker(files_chunk, model_id, device, torch_dtype, data_path, progress_queue):
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
        # model_kwargs={"model_max_length": 81810},
        # initial_prompt="Trascrizione di un'omelia registrata durante una messa:"
    )

    for file in files_chunk:
        process_file(file, pipe, data_path, progress_queue)


# def process_all_files_in_parallel(
#     files, model_id, device, torch_dtype, data_path, num_workers=4
# ):
#     chunk_size = len(files) // num_workers
#     chunks = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]

#     processes = []
#     for chunk in chunks:
#         p = mp.Process(
#             target=worker, args=(chunk, model_id, device, torch_dtype, data_path)
#         )
#         processes.append(p)
#         p.start()

#     for p in processes:
#         p.join()

def process_all_files_in_parallel(files, model_id, device, torch_dtype, data_path, num_workers=4):
    chunk_size = len(files) // num_workers
    chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]

    # Set up the manager and progress queue for multiprocessing
    with mp.Manager() as manager:
        progress_queue = manager.Queue()
        processes = []
        
        # Start a tqdm progress bar
        with tqdm(total=len(files)) as pbar:
            # Spawn worker processes
            for chunk in chunks:
                p = mp.Process(target=worker, args=(chunk, model_id, device, torch_dtype, data_path, progress_queue))
                processes.append(p)
                p.start()
            
            # Track progress in the main process
            processed_files = 0
            while processed_files < len(files):
                # Update progress bar when a worker notifies progress
                progress_queue.get()
                pbar.update(1)
                processed_files += 1
            
            # Wait for all processes to complete
            for p in processes:
                p.join()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    all_omelie = list(data_path.rglob("*melia*.mp3"))
    all_omelie += [
        f
        for f in data_path.rglob("*.mp3")
        if any([str(year) in str(f) for year in range(2018, 2025)])
    ]
    all_omelie = sorted(all_omelie)[::-1]

    to_process = [
        f
        for f in all_omelie
        if not (data_output / (get_output_filename(f, data_path)  + "text.txt")).exists()
    ]

    print(f"Found {len(all_omelie)} files, processing {len(to_process)} files...")

    start_time = time.time()

    num_workers = 3  # mp.cpu_count() // 2  # Use all available CPUs
    process_all_files_in_parallel(
        to_process, model_id, device, torch_dtype, data_path, num_workers=num_workers
    )

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time} seconds.")
