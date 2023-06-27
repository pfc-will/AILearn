from huggingface_hub import snapshot_download

def wrapper(repo_id: str, ignore_regex: list = []):
    try:
        snapshot_download(repo_id=repo_id, ignore_regex=ignore_regex)
    except:
        pass
    
# wrapper(repo_id="decapoda-research/llama-7b-hf", ignore_regex=["*.h5", "*.ot", "*.msgpack"])