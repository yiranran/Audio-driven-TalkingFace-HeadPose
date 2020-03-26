# Audio-driven Talking Face Video Generation with Learning-based Personalized Head Pose

We provide PyTorch implementations for our arxiv paper "Audio-driven Talking Face Video Generation with Learning-based Personalized Head Pose"(http://arxiv.org/abs/2002.10137).

Note that this code is protected under patent. It is for research purposes only at your university (research institution) only. If you are interested in business purposes/for-profit use, please contact Prof.Liu (the corresponding author, email: liuyongjin@tsinghua.edu.cn).

We provide a demo video [here](https://cg.cs.tsinghua.edu.cn/people/~Yongjin/Yongjin.htm) (please search for "Talking Face" in this page and click the "demo video" button).

## Our Proposed Framework

<img src = 'pipeline.jpg'>

## Prerequisites
- Linux or macOS
- NVIDIA GPU
- Python 3
- MATLAB

## Getting Started
### Installation
- You can create a virtual env, and install all the dependencies by
```bash
pip install -r requirements.txt
```

### Download pre-trained models
- Including pre-trained general models and models needed for face reconstruction, identity feature extraction etc
- Download from [BaiduYun](https://pan.baidu.com/s/1yAArGCkiKDICr0lM9U-_lw)(extract code:te4b) or [GoogleDrive](https://drive.google.com/open?id=1AwDmrSvmV7rktuWlptxVepxqDn8vTUp0) and copy to corresponding subfolders (Audio, Deep3DFaceReconstruction, render-to-video).

### Fine-tune on a target peron's short video
- 1. Prepare a talking face video that satisfies: 1) contains a single person, 2) 25 fps, 3) longer than 12 seconds, 4) without large body translation (e.g. move from the left to the right of the screen). An example is [here](Data/31.mp4). Rename the video to [person_id].mp4 (e.g. 1.mp4) and copy to Data subfolder.

Note: You can make a video to 25 fps by 
```bash
ffmpeg -i xxx.mp4 -r 25 xxx1.mp4
```
- 2. Extract frames and lanmarks by
```bash
cd Data/
python extract_frame1.py [person_id].mp4
```
- 3. Conduct 3D face reconstruction. First should compile code in `Deep3DFaceReconstruction/tf_mesh_renderer/mesh_renderer/kernels` to .so, following its [readme](Deep3DFaceReconstruction/tf_mesh_renderer/README.md), and modify line 28 in [rasterize_triangles.py](Deep3DFaceReconstruction/tf_mesh_renderer/mesh_renderer/rasterize_triangles.py) to your directory. Then run
```bash
cd Deep3DFaceReconstruction/
CUDA_VISIBLE_DEVICES=0 python demo_19news.py ../Data/[person_id]
```
This process takes about 2 minutes on a Titan Xp.
- 4. Fine-tune the audio network. First modify line 28 in [rasterize_triangles.py](Audio/code/mesh_renderer/rasterize_triangles.py) to your directory. Then run
```bash
cd Audio/code/
python train_19news_1.py [person_id] [gpu_id]
```
The saved models are in `Audio/model/atcnet_pose0_con3/[person_id]`.
This process takes about 5 minutes on a Titan Xp.
- 5. Fine-tune the gan network.
Run
```bash
cd render-to-video/
python train_19news_1.py [person_id] [gpu_id]
```
The saved models are in `render-to-video/checkpoints/memory_seq_p2p/[person_id]`.
This process takes about 40 minutes on a Titan Xp.


### Test on a target peron
Place the audio file (.wav or .mp3) for test under `Audio/audio/`.
Run
```bash
cd Audio/code/
python test_personalized.py [audio] [person_id] [gpu_id]
```
This program will print 'saved to xxx.mov' if the videos are successfully generated.
It will output 2 movs, one is a video with face only (_full9.mov), the other is a video with background (_transbigbg.mov).

## Acknowledgments
The face reconstruction code is from [Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction), the arcface code is from [insightface](https://github.com/deepinsight/insightface), the gan code is developed based on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).