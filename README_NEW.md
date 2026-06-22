# PIAD: Pose and Illumination Agnostic Anomaly Detection

## Setup

Clone the repository:

```shell
git clone https://github.com/VuHuyBui/piad_baseline.git
cd piad_baseline
```

Then download the pre-trained EfficientNet-B4 weights from the [PAD GitHub repository](https://github.com/EricLee0224/PAD) and place them in the corresponding location:

```shell
cd retrieval
gdown https://drive.google.com/uc?id=1CVFXJf203tDLKDNF2lx1FYNxWPNrZyt-
unzip model.zip
```

The resulting file structure should look like this:

```
retrieval
 └ model
```

You will also need the pre-trained weights for EfficientLoFTR. You can find the download link in the [EfficientLoFTR GitHub repository](https://github.com/zju3dv/EfficientLoFTR), or download `eloftr_outdoor.ckpt` directly from [this Google Drive folder](https://drive.google.com/drive/folders/1DaIBmLR_mPWJvTz3nkT890ytvDZwVMwV?usp=sharing). Place the file in `./EfficientLoFTR/weights/`.

The resulting file structure should look like this:

```
EfficientLoFTR
 └ weights
    └ eloftr_outdoor.ckpt
```

### Apptainer

To set up Apptainer, build the image with:

```shell
mkdir -p "$HOME/apptainer-tmp" "$HOME/apptainer-cache"
export APPTAINER_TMPDIR="$HOME/apptainer-tmp"
export TMPDIR="$HOME/apptainer-tmp"
export APPTAINER_CACHEDIR="$HOME/apptainer-cache"
apptainer cache clean -f
apptainer -d build piad.sif piad.def
```

Then open the shell environment with:

```shell
apptainer shell --nv piad.sif
```

## Building Custom Datasets

To run on a custom dataset:

1. Create a new file in the `datasets` folder and define your custom class, following the example in `custom.py` for the Cube dataset.
2. Create a corresponding config file in the `config/` folder (see `Cube.txt` for an example).

**Note:** `custom.py` defines the dataset following the `LEGO_3D.py` format, which expects the data folder to be structured as follows:

```
Cube
├ ground_truth
│  └ defect
├ test
│  ├ good
│  └ defect
├ train
│  └ good
└ transforms.json
```

`transforms.json` contains the 4x4 camera matrix for each image:

```json
{
    "camera_angle_x": 0.6911111611634243,
    "frames": [
        {
            "file_path": "./train/good/0.png",
            "transform_matrix": [
                [-0.8660253882408142, 0.49809738993644714, -0.04357787221670151, -0.15252254903316498],
                [-0.5000000596046448, -0.8627299070358276, 0.07547908276319504, 0.2641768157482147],
                [0.0, 0.08715573698282242, 0.9961947202682495, 3.4866814613342285],
                [0.0, 0.0, 0.0, 1.0]
            ]
        }
        // ...
    ]
}
```

## Running

### Pose Estimation

To run the optimizer:

```shell
python pose_estimation.py --config <path_to_config.txt> --class_name <class_name>
```

`pose_estimation.py` first generates a set of reflection images and trains 3DGS — these two steps only run on the initial pass. It then estimates the camera pose for each query image, renders the reference images, and saves the results to `./output`.

### Evaluation (Anomaly Detection)

To run anomaly detection:

```shell
python AUROC_TEST.py --obj <class_name>
```

Results are saved to `./AD_result`.