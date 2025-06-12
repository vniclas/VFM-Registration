import pickle
import pprint
from pathlib import Path
from time import time
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import rospy
import torch
import torch.nn.functional as F
from featup.featurizers.maskclip.clip import tokenize
from featup.util import pca
from numpy.typing import ArrayLike
from pytorch_lightning import seed_everything
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

pp = pprint.PrettyPrinter(indent=4)

seed_everything(0)


class ImageFeatureGenerator:

    def __init__(
        self,
        foundation_model: str,
        use_featup: bool = True,
    ) -> None:
        self.foundation_model_name = foundation_model
        self.use_featup = use_featup

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.patch_h = 20
        self.patch_h = 16  # -> same as featup
        self.patch_w = None

        if self.foundation_model_name == "dinov2":
            self.model = torch.hub.load("mhamilton723/FeatUp",
                                        "dinov2",
                                        use_norm=True,
                                        trust_repo=True).eval().to(self.device)
            self.patch_size = 14
            self.feature_size = 384
        elif self.foundation_model_name == "maskclip":
            self.model = torch.hub.load("mhamilton723/FeatUp",
                                        "maskclip",
                                        use_norm=False,
                                        trust_repo=True).eval().to(self.device)
            self.patch_size = 16
            self.feature_size = 512
        else:
            rospy.logerr(f"Unsupported foundation model: {foundation_model}")
            raise ValueError(f"Unsupported foundation model: {foundation_model}")

        self.transform = None
        self.image_shape = [-1, -1]

        self.fit_pca = {}
        self.fit_pca_file = Path(__file__).parent / "pca_fit.pkl"
        if self.fit_pca_file.exists():
            with open(self.fit_pca_file, 'rb') as f:
                self.fit_pca = pickle.load(f)

        rospy.loginfo(f"Model loaded: {self.foundation_model_name}")

    def create_transform_(self, img_h, img_w) -> None:
        scale = (self.patch_size * self.patch_h) / img_h
        self.patch_w = int(scale * img_w / self.patch_size)

        self.transform = Compose([
            ToTensor(),  # HxWxC with [0, 255] -> CxHxW [0.0, 1.0]
            Resize((self.patch_size * self.patch_h, self.patch_size * self.patch_w),
                   antialias=False),  # Default is bilinear
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_shape = (img_h, img_w)

    def get_image_features(self,
                           image: ArrayLike,
                           upsample: bool = False,
                           cache_file: str = "") -> ArrayLike:
        # Load cache if available
        features = None
        if cache_file:
            cache_file = cache_file.parent / f"{cache_file.stem}_{self.use_featup}_{upsample}.npy"
            if cache_file.exists():
                features = np.load(cache_file, allow_pickle=True)

        if features is None:
            if self.image_shape[0] != image.shape[0] or self.image_shape[1] != image.shape[1]:
                self.create_transform_(image.shape[0], image.shape[1])

            # Pre-process image and add batch dimension
            torch_image = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if self.use_featup:
                    features = self.model(torch_image)
                else:
                    features = self.model.model(torch_image)
            # Feature size: BxCxHxW

            if upsample:
                features = F.interpolate(features,
                                         image.shape[:2],
                                         mode="bilinear",
                                         align_corners=False)

            features = features.squeeze().permute(1, 2, 0).cpu().numpy()

            # Save cache
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_file, features)

        return features

    def get_image_features_pca(self,
                               image: ArrayLike,
                               upsample: bool = False,
                               n_components: int = 3) -> Tuple[ArrayLike, ArrayLike]:
        if self.image_shape[0] != image.shape[0] or self.image_shape[1] != image.shape[1]:
            self.create_transform_(image.shape[0], image.shape[1])

        # Pre-process image and add batch dimension
        torch_image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.use_featup:
                features = self.model(torch_image)
            else:
                features = self.model.model(torch_image)
        # Feature size: BxCxHxW

        [pca_features], fit_pca = pca([features],
                                      dim=n_components,
                                      fit_pca=self.fit_pca.get(n_components, None))
        # Only fit PCA once
        if n_components not in self.fit_pca:
            self.fit_pca[n_components] = fit_pca

        if upsample:
            features = F.interpolate(features,
                                     image.shape[:2],
                                     mode="bilinear",
                                     align_corners=False)
            if pca_features is not None:
                pca_features = F.interpolate(pca_features,
                                             image.shape[:2],
                                             mode="bilinear",
                                             align_corners=False)

        features = features.squeeze().permute(1, 2, 0).cpu().numpy()

        pca_features = pca_features.permute(0, 2, 3, 1).squeeze().cpu().numpy()
        if n_components == 3:
            pca_features = (pca_features * 255.0).astype(np.uint8)  # Convert to RGB

        return features, pca_features

    def run_pca(self,
                features: ArrayLike,
                refit_pca: bool = False,
                n_components: int = 3) -> ArrayLike:

        features_torch = torch.from_numpy(features.T).to(device=self.device)
        features_torch = features_torch.unsqueeze(0).unsqueeze(-1)

        if refit_pca:
            self.fit_pca[n_components] = None

        [pca_features], fit_pca = pca([features_torch],
                                      dim=n_components,
                                      fit_pca=self.fit_pca.get(n_components, None))
        # Only fit PCA once
        if n_components not in self.fit_pca:
            self.fit_pca[n_components] = fit_pca
            with open(self.fit_pca_file, 'wb') as f:
                pickle.dump(self.fit_pca, f)

        pca_features = pca_features.permute(0, 2, 3, 1).squeeze().cpu().numpy()

        if n_components == 3:
            pca_features = (pca_features * 255.0).astype(np.uint8)  # Convert to RGB

            # Set RGB to black for points w/o features
            pca_features[np.all(features == 0,
                                axis=-1)] = np.zeros_like(pca_features[np.all(features == 0,
                                                                              axis=-1)])

        return pca_features

    def compute_similarity(self, features: ArrayLike, prompt: str) -> ArrayLike:
        with torch.no_grad():
            tokens = tokenize(prompt).to(self.device)
            embedding = self.model.model.model.encode_text(tokens)
            embedding = embedding.detach().cpu().numpy().T

        embedding_norm = embedding / np.linalg.norm(embedding)
        features_norm = np.zeros_like(features, features.dtype)
        non_zero_idx = np.all(features != 0, axis=-1)
        features_norm[non_zero_idx] = features[non_zero_idx] / np.linalg.norm(
            features[non_zero_idx], axis=-1)[:, None]

        similarities = np.dot(features_norm, embedding_norm)

        return similarities
