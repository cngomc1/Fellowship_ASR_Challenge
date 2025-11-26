from huggingface_hub import snapshot_download

def download_dataset():
    snapshot_download(
        repo_id="DigitalUmuganda/ASR_Fellowship_Challenge_Dataset",
        repo_type="dataset",
        local_dir="data"
    )

if __name__ == "__main__":
    download_dataset()
