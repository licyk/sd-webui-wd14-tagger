import json
import copy
from typing import Tuple, Dict, Union
from huggingface_hub import hf_hub_download
from PIL import Image
from timm import create_model
import torch
from torchvision.transforms import Compose
import pandas as pd
# from imgutils.preprocess import create_torchvision_transforms

from tagger.interrogator import Interrogator


class ConvnextInterrogator(Interrogator):
    def __init__(
        self,
        name: str,
        model_path="model.safetensors",
        tag_mapping_path="selected_tags.csv",
        preprocess_path="preprocess.json",
        **kwargs,
    ) -> None:
        super().__init__(name)
        self.model_path = model_path
        self.tag_mapping_path = tag_mapping_path
        self.preprocess_path = preprocess_path
        self.kwargs = kwargs

    def download(self) -> Tuple[Compose, torch.nn.Module, pd.DataFrame]:
        print(f"Loading {self.name} model file from {self.kwargs['repo_id']}")

        with open(
            hf_hub_download(
                **self.kwargs,
                filename=self.preprocess_path,
            ),
            encoding="utf-8",
        ) as f:
            preprocess = create_torchvision_transforms(json.load(f)["test"])
        model = create_model(f"hf-hub:{self.kwargs['repo_id']}", pretrained=True)
        df_tags = pd.read_csv(
            hf_hub_download(
                **self.kwargs,
                filename=self.tag_mapping_path,
            )
        )
        return preprocess, model, df_tags

    def load(self) -> None:
        preprocess, model, df_tags = self.download()

        self.model = model
        self.preprocess = preprocess
        self.df_tags = df_tags

        print(f"Loaded {self.name} model")

    def interrogate(
        self, image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float],  # tag confidents
    ]:
        # init model
        if not hasattr(self, "model") or self.model is None:
            self.load()

        input_ = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_)
            prediction = torch.sigmoid(output)[0]

        tags = self.df_tags["name"]
        mask = prediction.numpy() >= self.df_tags["best_threshold"]
        res = dict(zip(tags[mask].tolist(), prediction[mask].tolist()))

        rating_type = ["general", "sensitive", "questionable", "explicit"]
        ratings = {k: res[k] for k in rating_type if k in res}
        for t in rating_type:
            res.pop(t, None)

        return ratings, res

    def unload(self) -> bool:
        unloaded = False

        if hasattr(self, "model") and self.model is not None:
            del self.model
            unloaded = True
            print(f"Unloaded {self.name}")

        if hasattr(self, "df_tags"):
            del self.df_tags

        if hasattr(self, "preprocess"):
            del self.preprocess

        return unloaded
