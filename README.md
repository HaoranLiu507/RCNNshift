# RCNNshift for Moving Object Tracking (PyTorch)

GPU-accelerated implementation of RCNNshift for single-object tracking. RCNNshift augments the classic MeanShift tracker with features produced by a Random‑coupled Neural Network (RCNN) or its temporal variant (3D‑RCNN). Features are generated per frame as an ignition map and fused with image channels to perform robust tracking in an enhanced feature space. A baseline OpenCV MeanShift tracker is also provided.

![Demo](demo.gif)

---

## Features
- **RCNNshift (2D)**: Frame-wise RCNN ignition features + MeanShift.
- **RCNNshift_3D (temporal)**: 3D‑RCNN over a temporal window to inject short-term motion context.
- **OpenCV MeanShift baseline**: Quick comparison in HSV color space.
- **GPU acceleration**: PyTorch kernels with batch processing for speed.
- **Interactive or scripted ROI selection**: Mouse selection, manual input, or batch string.

---

## Requirements
- Python 3.11 (newer versions likely work)
- NVIDIA GPU with CUDA 11.8 recommended
- PyTorch 2.1.0 with matching CUDA build
- OpenCV, NumPy, tqdm

Install dependencies:

```bash
# 1) Install PyTorch that matches your CUDA (example: CUDA 11.8)
# See: https://pytorch.org/get-started/locally/
pip3 install torch --index-url https://download.pytorch.org/whl/cu118

# 2) Install the rest
pip install -r requirements.txt
```

The development environment used: Python 3.11, CUDA 11.8, Torch 2.1.0.

> Note: The current default device is GPU. To force CPU, change the device in `RCNNshift.py`:
>
> ```python
> # In RCNNshift.py (top of file)
> device = torch.device("cuda:0")
> # device = torch.device("cpu")
> ```
>
> On CPU you must also change tensor dtypes from half precision to float32 in `RCNNshift.py` (search for `torch.float16` and replace with `torch.float32`). PyTorch does not support conv2d/conv3d in float16 on CPU. GPU is recommended.

---

## Dataset
This repo demonstrates tracking on the TB‑50 benchmark. The dataset is not included. You can download a prepared copy here: [Zenodo DOI for TB‑50](https://doi.org/10.5281/zenodo.12526486).

- Place your videos (MP4) in a folder and point `main.py` to it via `path` and `name` (file name without extension).
- The repo includes a `GroundTruth/` directory with sample bounding boxes for common sequences for reference/comparison.

---

## Quickstart
1) Clone this repository and install requirements (see above).

2) Put an `.mp4` video under a directory of your choice (e.g., `videos/blurbody.mp4`).

3) Edit `main.py` to point to your video and choose a tracker and options:

```python
# main.py
import RCNNshift as tracker
import os

# Video path and name
path = os.path.abspath('videos')     # directory containing your .mp4
name = 'blurbody'                    # filename WITHOUT extension

# Tracker options: 'RCNNshift', 'RCNNshift_3D', or 'meanshift'
select_tracker = 'RCNNshift'

# ROI selection: 'mouse' (interactive), 'input' (type x y w h), or 'batch_input' (pass string)
select_rect = 'mouse'

# Runtime: 'live' (show while tracking) or 'local' (save and display at the end)
perform = 'live'

Tracker = tracker.RCNNshift(
    weight=38,
    batch_size=50,
    isColor=True,          # for 3D tracker: True for RGB videos, False for grayscale
    select_tracker=select_tracker,
    perform=perform,
    depth=3                # temporal window size for 'RCNNshift_3D'
)

Tracker.track(video_path=path, name=name, select_rect=select_rect, ROI_region=None)
```

4) Run:

```bash
python main.py
```

5) Select the initial ROI in the first frame:
- **mouse**: Click and drag to draw the box, then press any key (e.g., Enter) to continue.
- **input**: Type `x y w h` into the terminal when prompted (e.g., `252 160 68 98`).
- **batch_input**: Provide `ROI_region='x y w h'` when calling `track()`.

---

## Outputs
- Per-frame track windows are saved as text to:
  - `TrackWindowResult/RCNNshift/<name>.txt` for RCNNshift
  - `TrackWindowResult/RCNNshift_3D/<name>.txt` for RCNNshift_3D
  - `TrackWindowResult/Meanshift/<name>.txt` for baseline
  Each line is `x,y,w,h` for a frame.
- If `perform='local'`, a rendered video is saved to:
  - `TrackedVideo/<TrackerName>/<name>.mp4`

---

## Configuration reference
- **select_tracker**: `'RCNNshift' | 'RCNNshift_3D' | 'meanshift'`
  - `RCNNshift_3D` adds short temporal context and can improve robustness on fast motion; it is slightly heavier.
- **select_rect**: `'mouse' | 'input' | 'batch_input'`
  - For `input` and `batch_input`, provide bounding boxes as `x y w h` in pixel units.
- **perform**: `'live' | 'local'`
  - `live` shows frames while tracking. `local` writes results first, then renders one video at the end.
- **weight**: Controls the contribution of the ignition feature channel during quantization for histogramming. Typical values: `20–60` (default used in examples: `38`). Larger values emphasize ignition more and increase the number of histogram bins internally.
- **batch_size**: Number of frames processed per batch by RCNN/3D‑RCNN. Increase for better throughput if you have VRAM headroom; decrease if you hit OOM.
- **isColor**: For `RCNNshift_3D`, set `True` for RGB videos, `False` for grayscale.
- **depth**: Temporal window for 3D‑RCNN (odd integer, e.g., `3`, `5`). `3` is a good default.

Performance tips:
- Out‑of‑memory (OOM) on GPU: lower `batch_size` and/or `depth` (for 3D), and close other GPU apps.
- Tracking stability: try a slightly larger `weight` if the target is weak in RGB but strong in ignition; reduce `weight` if it drifts to high‑ignition clutter.

---

## How it works (high level)
- RCNN/3D‑RCNN generates an ignition map for each frame (or temporal window).
- The ignition map is fused with RGB to create a 4‑channel representation.
- Quantized histograms use the R channel together with the ignition channel from the 4‑channel image to drive MeanShift iterations.

For deeper details and evaluation, see the forthcoming paper (DOI: to be announced).

---

## Troubleshooting
- "Video file does not exist: ...": ensure `path` points to the directory containing your `name.mp4`, and that the file exists.
- `ModuleNotFoundError: No module named 'tqdm'`: `pip install tqdm`.
- No GPU available: either install a CUDA build of PyTorch for your GPU, or switch to CPU by editing `device` in `RCNNshift.py` (see snippet above) and changing tensor dtypes to float32 (conv ops do not support float16 on CPU). CPU will be significantly slower.
- Windows display issues: if interactive windows do not appear, make sure no other OpenCV windows are blocking input. Press `Esc` (RCNNshift) or `q` (MeanShift/show) to stop.

---

## What's in this repo
- `main.py`: Entry point. Reads video, sets parameters, launches tracking.
- `RCNNshift.py`: RCNN/3D‑RCNN feature extraction and MeanShift‑based tracking logic.
- `GroundTruth/`: Example ground‑truth boxes for common sequences (for reference).
- `demo.gif`: Short visualization of the tracker.
- `requirements.txt`: Minimal runtime dependencies (install PyTorch separately as shown above).
- `LICENSE`: MIT.

> Note: Earlier drafts mentioned `main_track_in_batch.py` for batch evaluation. This file is not included in this repository. If you need batch evaluation scripts, please open an issue.

---

## License
This project is released under the MIT License. See `LICENSE` for details.

## Citation
If you use this code in your research, please cite this repository. A formal citation will be added once the paper is available.

```
@software{RCNNshift,
  author  = {Liu, Haoran},
  title   = {RCNNshift: GPU-accelerated Random-coupled Neural Network + MeanShift tracking},
  year    = {2025},
  url     = {https://github.com/your-org-or-user/RCNNshift}
}
```

## Contact
Questions or issues? Please use GitHub Issues or email: liuhaoran@cdut.edu.cn


