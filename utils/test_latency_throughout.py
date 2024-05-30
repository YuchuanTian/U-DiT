# Copyright 2024 Huawei Technologies Co., Ltd
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
# ============================================================================
import os
import time
import torch
from tqdm import tqdm
import warnings
import numpy as np
import torch.nn as nn
from torchprofile import profile_macs

warnings.filterwarnings('ignore')

from ..udit_models import DiT_models as models

def test_flops_params(test_models):
    inputs = torch.rand(1, 4, 32, 32).cuda()
    t = torch.ones(1).int().cuda()
    y = torch.ones(1).int().cuda()
    for model_name in list(test_models):
        model = models[model_name]()
        count = 0
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                count += 1
        model.cuda()
        flops = profile_macs(model, (inputs, t, y))              
        params = np.sum([x.numel() for x in model.parameters()]) 
        print(model_name+f', Conv Layer = {count}, FLOPs = ' + str(flops / 1000 ** 3) + ' G' + ', Params = ' + str(params / 1000 ** 2) + ' M')
    

def test_latency(test_models):
    '''
        bs=1, latency
    '''
    inputs = torch.rand(1, 4, 32, 32).cuda()
    t = torch.ones(1).int().cuda()
    y = torch.ones(1).int().cuda()
    for model_name in list(test_models):
        model = models[model_name]()
        model.cuda()
        # warm up
        with torch.no_grad():
            for _ in range(5):
                out = model(inputs, t, y)
        iters = 20
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iters):
                out = model(inputs, t, y)
        end_time = time.time()
        duration = (end_time-start_time)/iters
        print(model_name+", Inference = " +f"{duration:.4f} seconds")

def model_infer(model, bs):
    inputs = torch.rand(bs, 4, 32, 32).cuda()
    t = torch.ones(1).int().cuda()
    y = torch.ones(1).int().cuda()
    try:
        with torch.no_grad():
            for _ in range(2):
                model(inputs, t, y)
        print(f"==> bs={bs} is ok")
        return True
    except:
        print(f"==> bs={bs} is too large")
        return False

def binarySearch(min_bs, max_bs, model): 
    if model_infer(model, max_bs):
        return max_bs
    elif (max_bs-min_bs)<=1:
        return min_bs
    else:
        mid_bs = (min_bs+max_bs)//2
        if model_infer(model, mid_bs):
            return binarySearch(mid_bs, max_bs, model)
        else:
            return binarySearch(min_bs, mid_bs, model)
    
def test_throughout(test_models):
    '''
        maximum batch to 32 G GPU memory 
    '''
    for model_name in list(test_models):
        torch.cuda.empty_cache()
        print("\n"+"="*40+model_name+"="*40)
        model = models[model_name]()
        model.cuda()
        t = torch.ones(1).int().cuda()
        y = torch.ones(1).int().cuda()
        # test maxmimum batch size
        print("Binary searching......")
        bs = binarySearch(0, 70000, model)
        print(f"search finished, max batch size = {bs}")
        torch.cuda.empty_cache()
        inputs = torch.rand(bs, 4, 32, 32).cuda()
        t = torch.ones(1).int().cuda()
        y = torch.ones(1).int().cuda()
        # warm up
        print(f"warm up")
        torch.cuda.empty_cache()
        with torch.no_grad():
            for _ in tqdm(range(2)):
                model(inputs, t, y)
        iters = 10
        print(f"start test throughout")
        torch.cuda.empty_cache()
        start_time = time.time()
        with torch.no_grad():
            for _ in tqdm(range(iters)):
                model(inputs, t, y)
        end_time = time.time()
        throughout = (iters*bs)/(end_time-start_time)
        print(f"{model_name}, throughout = {throughout:.2f} samples/second")

def test_model_throughout(model, model_name='', bs=None):
    '''
        maximum batch to 32 G GPU memory 
    '''
    torch.cuda.empty_cache()
    model.cuda()
    t = torch.ones(1).int().cuda()
    y = torch.ones(1).int().cuda()
    # test maxmimum batch size
    if bs is None:
        print("Binary searching......")
        bs = binarySearch(0, 70000, model)
    print(f"search finished, max batch size = {bs}")
    torch.cuda.empty_cache()
    inputs = torch.rand(bs, 4, 32, 32).cuda()
    t = torch.ones(1).int().cuda()
    y = torch.ones(1).int().cuda()
    # warm up
    print(f"warm up")
    torch.cuda.empty_cache()
    with torch.no_grad():
        for _ in tqdm(range(10)):
            model(inputs, t, y)
    iters = 10
    print(f"start test throughout")
    torch.cuda.empty_cache()
    start_time = time.time()
    with torch.no_grad():
        for _ in tqdm(range(iters)):
            model(inputs, t, y)
    end_time = time.time()
    throughout = (iters*bs)/(end_time-start_time)
    print(f"{model_name}, throughout = {throughout:.2f} samples/second")

def main():
    test_models = ["IPDTv5-S"]
    test_throughout(test_models)

if __name__=="__main__":
    main()