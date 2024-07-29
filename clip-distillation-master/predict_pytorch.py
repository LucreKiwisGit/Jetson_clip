# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
python3 predict_pytorch.py \
    resnet18 \
    data/models/resnet18/checkpoint.pth \
    data/text_embeddings.npy \
    assets/cat.jpg \
    --text_prompts data/text_prompts.txt
"""

import timm
import torch
import PIL.Image
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import open_clip
import time
from open_clip.pretrained import _PRETRAINED

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def embedding_to_probs(embedding, text_embedding, temp=100.):
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    logits = embedding @ text_embedding.T
    logits = F.softmax(temp * logits, dim=-1)
    return logits


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("text_embeddings_path", type=str)
    parser.add_argument("image_path", type=str)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--output_dim", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--text_prompts", type=str, default=None)
    parser.add_argument("--use_asp", action="store_true")
    parser.add_argument("--use_qat", action="store_true")
    args = parser.parse_args()

    if args.text_prompts is not None:

        with open(args.text_prompts, 'r') as f:
            text_prompts = f.readlines()
            text_prompts = [tp.strip() for tp in text_prompts]

        print(f"Found the following {len(text_prompts)} text prompts in {args.text_prompts}")
        print(text_prompts)
    else:
        text_prompts = None

    if args.use_qat:
        from pytorch_quantization import quant_modules
        # use QAT monkey-patching
        print("Initializing quantization aware training (QAT)")
        quant_modules.initialize()

    # distilled model
    model_distill = timm.create_model(
        model_name=args.model_name,
        num_classes=args.output_dim,
    )
    checkpoint = torch.load(args.checkpoint_path)
    model_distill.load_state_dict(checkpoint["model"])

    # teacher model
    model_name = "ViT-B-32"
    model_teacher, _, preprocess = open_clip.create_model_and_transforms(
        model_name, 
        pretrained="../PretrainedModel/openclip/ViT-B-32/open_clip_pytorch_model.bin"
    )

    model_teacher = model_teacher.to(args.device).eval()
    model_distill = model_distill.to(args.device).eval()

    if args.use_asp:
        from apex.contrib.sparsity import ASP
        ASP.init_model_for_pruning(model_distill, mask_calculator="m4n2_1d", verbosity=2, whitelist=[torch.nn.Linear, torch.nn.Conv2d], allow_recompute_mask=False, allow_permutation=False)
        # ASP.init_optimizer_for_pruning(optimizer)
        ASP.compute_sparse_masks()

    

    transform = Compose([
        Resize(args.image_size),
        CenterCrop(args.image_size),
        ToTensor(),
        Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    image = PIL.Image.open(args.image_path).convert("RGB")

    text_embeddings = torch.from_numpy(
        np.load(args.text_embeddings_path)
    ).to(args.device).float()

    with torch.no_grad():
        image_data = transform(image).to(args.device)

        # warmup
        for i in range(3):
            output_embedding_distill = model_distill(image_data[None, ...]) 
            output_embedding_teacher = model_teacher.encode_image(image_data[None, ...])

        start_time = time.time()
        output_embedding_distill = model_distill(image_data[None, ...]) 
        probs_distill = embedding_to_probs(
            output_embedding_distill,
            text_embeddings
        )
        probs_distill = probs_distill.detach().cpu().numpy()
        inference_distill_time = time.time() - start_time

        output_embedding_teacher = model_teacher.encode_image(image_data[None, ...])
        probs_teacher = embedding_to_probs(
            output_embedding_teacher,
            text_embeddings
        )
        probs_teacher = probs_teacher.detach().cpu().numpy()
        inference_teacher_time = time.time() - start_time - inference_distill_time

    probs_distill = probs_distill.flatten()
    prob_indices_distill = np.argsort(probs_distill)[::-1] # descending

    probs_teacher = probs_teacher.flatten()
    probs_indices_teacher = np.argsort(probs_teacher)[::-1]

    print(f"Result of {args.model_name}, it takes {inference_distill_time * 1000.0} ms:")
    if text_prompts is not None:
        for pid in prob_indices_distill:
            print(f"Index {pid} ({100 * round(probs_distill[pid], 3)}%): \"{text_prompts[pid]}\"")
    else:
        for pid in prob_indices_distill:
            print(f"Index {pid} ({100 * round(probs_distill[pid], 3)}%)")

    print(f"Result of ViT-B-32, it takes {inference_teacher_time * 1000.0} ms:")
    if text_prompts is not None:
        for pid in probs_indices_teacher:
            print(f"Index {pid} ({100 * round(probs_teacher[pid], 3)}%): \"{text_prompts[pid]}\"")
    else:
        for pid in probs_indices_teacher:
            print(f"Index {pid} ({100 * round(probs_teacher[pid], 3)}%)")