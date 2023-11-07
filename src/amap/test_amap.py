import sys
sys.path.append('../')
import torch
from datasets import Dataset
from src.model.causal_lm import CausalLanguageModel
from src.data.dataset_loader import load_hf_dataset_with_sampling
from src.amap.amap import LMamap

def test_amap_extraction(device, fp16=False):

    model_name = "EleutherAI/pythia-70m-deduped"
    window_size = 15
    window_stride = 15
    n_samples = 3
    batch_size = 1

    model = CausalLanguageModel(model_name, device="cuda" if torch.cuda.is_available() and device=="cuda" else "cpu", fp16=fp16)

    mode = ['input', 'output']

    amapper = LMamap(model=model,
                     device=device,
                     mode=mode,
                     fp16=fp16)

    amapper.add_position(window_size)

    dataset = load_hf_dataset_with_sampling("wikipedia,20220301.en,train", n_samples=n_samples)

    amap, tokens_count, n_sentences = amapper.extract(
                dataset=dataset,
                batch_size=batch_size,
                window_size=window_size,
                window_stride=window_stride)

    # safety check
    warning_flag = amapper.sanity_check(n_sentences)

    print("[TEST] Result of the test: ", warning_flag)
    if warning_flag == '': print("[TEST] Sucessfully passed :-)")
    else: print("[TEST] Test failed")