# funky_boulder
Bouldering videos made funny


### Building tips
Install [Co-Tracker](https://github.com/facebookresearch/co-tracker)
```bash
git clone https://github.com/facebookresearch/co-tracker
cd co-tracker
python -m pip install -e .
```

Build decord with GPU support.
[Build with CMAKE](https://github.com/dmlc/decord/issues/19)
```bash
sudo apt-get install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavfilter-dev

cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release -DFFMPEG_DIR:PATH="/Data/miniconda3/envs/gpu/bin/ffmpeg"
```


