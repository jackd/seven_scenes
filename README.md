Python processing scripts for Microsoft's [seven scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) rgbd dataset.

# Setup
1. Clone this repository and add the parent directory to your `PYTHONPATH`
```
cd /path/to/parent_dir
git clone https://github.com/jackd/seven_scenes.git
export PYTHONPATH=$PYTHONPATH:/path/to/parent_dir
```
2. Download seven scenes data from the [official website](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
3. Set the `SEVEN_SCENES_PATH` environment variable to the parent directory of the downloaded zips
```
export SEVEN_SCENES_PATH=/path/to/downloaded_zips
```
4. Run `scripts/extract_scene_data.py` (or extract manually).

# Reconstructed Depth Data
Follow the instructions [here](preprocessed/README.md) for reconstructed depth data from the TSDF. While I originally intended to recreate this data from the base dataset, I've had some problems aligning the point clouds. See `example/ground_truth.py` for progress. Pull requests welcome.

In the meantime, fixed resolution data is available here thanks to [this work](https://bitbucket.org/shenlongwang/proximalnet/src/master/).
