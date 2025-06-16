Training code for CNN with PyTorch.


#### Installation

On my Linux box with NVIDIA gfx card (your setup may vary):

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

#### Running pipeline

- Go to `cmd/mlproducer` and do `go build`
- Collect lots of data with autoplay and copy /tmp/autoplay.txt somewhere
- Activate your venv then `cat /tmp/autoplay.txt | ../cmd/mlproducer/mlproducer | python3 training.py`
- You can pipe in `pv -br` after the call to `mlproducer` to see what the byte rate is. I see like 700 MiB on my computer. It can probably be made faster with more efficient `mlproducer` code.
- Wait a long time (maybe not that long, depends on how many games you used).
- Profit!