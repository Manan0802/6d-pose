# Dummy Dataset Loader
import time

class LinemodDataset:
    def __init__(self, split='train'):
        self.split = split
        # print(f"Initializing dummy Linemod dataset for {self.split} split...")
        time.sleep(0.1) # Simulate init
        self.data = ["dummy_item_1", "dummy_item_2", "dummy_item_3"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(f"Loading item {idx}...")
        return {"image": "dummy_image_data", "pose": "dummy_pose_data"}

# print("Dummy LinemodDataset class loaded.")
