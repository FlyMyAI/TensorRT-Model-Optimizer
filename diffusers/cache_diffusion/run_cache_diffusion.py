# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import time
from pathlib import Path

import torch

from cache_diffusion import cachify
from cache_diffusion.utils import SDXL_DEFAULT_CONFIG
from diffusers import DiffusionPipeline
from pipeline.deploy import compile, teardown

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe = pipe.to("cuda")

num_inference_steps = 4
prompt = "A random person with a head that is made of flowers, photo by James C. Leyendecker, \
    Afrofuturism, studio portrait, dynamic pose, national geographic photo, retrofuturism, \
        biomorphicy"

compile(
    pipe.unet,
    onnx_path=Path("./onnx"),
    engine_path=Path("./engine"),
)
cachify.prepare(pipe, num_inference_steps, SDXL_DEFAULT_CONFIG)

generator = torch.Generator(device="cuda").manual_seed(2946901)

start_time = time.time()

img = pipe(
    prompt=prompt,
    num_inference_steps=num_inference_steps,
    generator=generator,
    width=512,
    height=512
).images[0]

end_time = time.time()
inference_time = end_time - start_time

print(f"Time: {inference_time:.4f} seconds")

img.save("./test.png")
teardown(pipe.unet)
