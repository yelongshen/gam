from huggingface_hub import snapshot_download
import os

target_dir = "/home/grease/egodata/downloads/testset/PHUMA"
os.makedirs(target_dir, exist_ok=True)

print("Attempting to download PHUMA dataset from HuggingFace...")
try:
    # Try downloading the dataset. If it gives a 401 Unauthorized, it requires token access.
    snapshot_download(
        repo_id="DAVIAN-Robotics/PHUMA", 
        repo_type="dataset", 
        local_dir=target_dir, 
        max_workers=8,
        # We allow it to fetch even if large
    )
    print(f"Successfully downloaded PHUMA to {target_dir}")
except Exception as e:
    print("\nDownload failed with Exception:")
    print(e)
    print("\nThis likely means the dataset is gated or private, requiring a HF_TOKEN.")
