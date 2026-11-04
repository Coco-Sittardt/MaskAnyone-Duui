import os
import pickle
import requests
import json
import io
import numpy as np

class Sam2Client:
    _base_path: str

    def __init__(self, base_path: str):
        self._base_path = base_path
        # // make ping request to check if the server is reachable
        try:
            response = requests.get(self._make_url("ping"))
            if response.status_code != 200:
                raise ConnectionError(f"Error: Unable to reach SAM2 server at {self._base_path}. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error: Unable to reach SAM2 server at {self._base_path}. Exception: {e}")

    def segment_video(self, pose_prompts, video_content):
        files = {
            'video': ('video.mp4', video_content, 'video/mp4'),
        }

        data = {
            'pose_prompts': json.dumps(pose_prompts),
        }

        print(f"Sending request to {self._make_url('segment-video')} with pose_prompts: {pose_prompts}")

        response = requests.post(
            self._make_url("segment-video"),
            files=files,
            data=data,
        )

        print(f"Received response with status code: {response.status_code}")

        return response.content

    def decode_mask_npz_content(self, content):
        buffer = io.BytesIO(content)
        loaded = pickle.load(buffer)

        masks = {}
        for key in loaded.files:
            frame, mask = key.split('_mask')
            frame = int(frame.replace('frame', ''))
            mask = int(mask)
            if frame not in masks:
                masks[frame] = {}
            masks[frame][mask] = loaded[key]

        return masks

    def _make_url(self, path: str) -> str:
        return self._base_path + "/" + path
