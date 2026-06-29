# SLCalibration — Structured Light System Calibration via DIC

[**中文说明**](#中文说明) | [**English**](#english)

基于数字图像相关(DIC)的结构光系统标定方法。本代码为论文 **"Calibration Method for Structured Light System Based on DIC"** (Applied Optics, Vol. 61, No. 27, 2022) 中的算法。

该方法使用一个散斑投影仪和一台相机，通过 DIC 匹配实现 单目结构光的标定，不需要额外投影图案。

---

## 中文说明

### 算法原理

整个标定流程分为四个步骤:

```
标定板图像 → [圆心检测] → 相机图圆心坐标
                                  ↓
 相机图 + 散斑参考图 → [DIC匹配] → 散斑图中对应坐标
              ↓
 圆心坐标(左) + 匹配坐标(右) + 世界坐标 → [双目标定] → 内外参数
                                          ↓
                            已知匹配点对 → [三维重建] → 点云
```

使用 DIC 算法，将标定板圆心精确匹配到散斑图像中，从而将投影仪"看作"第二台相机进行立体标定。

### 环境要求

- Python 3.9+
- NumPy, OpenCV, SciPy, Matplotlib

```bash
pip install -r requirements.txt
```

### 数据准备

按以下结构组织数据:

```
your_data_dir/
├── board_pics/              # 标定板图像 (相机拍摄)
│   ├── 1.bmp                # 位姿1 (带散斑)
│   ├── 1off.bmp             # 位姿1 (无散斑, 可选)
│   ├── 2.bmp
│   ├── 2off.bmp
│   └── ...
├── speckle_pic/             # 散斑参考图像
│   └── speckle.bmp
├── referencecenters.csv     # 标定板圆点世界坐标 (Nx3)
├── cl.csv                   # 待重建的左图匹配点 (仅recon模式需要)
└── cr.csv                   # 待重建的右图匹配点 (仅recon模式需要)
```

支持的图像格式: `.bmp`, `.png`, `.jpg`, `.tif`

### 快速开始

```bash
# 方式1: 一键运行全流程
python pipeline.py --data_dir ./your_data_dir

# 方式2: 分步运行
python pipeline.py --data_dir ./your_data_dir --mode detect   # 步骤1: 圆心检测
python pipeline.py --data_dir ./your_data_dir --mode match    # 步骤2: DIC匹配
python pipeline.py --data_dir ./your_data_dir --mode calib    # 步骤3: 双目标定
python pipeline.py --data_dir ./your_data_dir --mode recon    # 步骤4: 三维重建

# 方式3: 使用配置文件
python pipeline.py --config ./config.json
```

### 命令行参数

| 参数                | 默认值     | 说明                                 |
| ----------------- | ------- | ---------------------------------- |
| `--data_dir`      | `.`     | 数据根目录                              |
| `--mode`          | `all`   | 运行模式: all/detect/match/calib/recon |
| `--n_images`      | `5`     | 标定板图像数量                            |
| `--board_width`   | `5`     | 标定板圆点列数                            |
| `--board_height`  | `3`     | 标定板圆点行数                            |
| `--symmetric`     | `False` | 是否对称标定板                            |
| `--r`             | `30`    | DIC子区半径 (像素)                       |
| `--search_radius` | `200`   | DIC搜索半径 (像素)                       |
| `--config`        | —       | JSON配置文件路径                         |
| `--img_ext`       | `bmp`   | 图像文件扩展名                            |

### 配置文件示例

```json
{
  "data_dir": "./my_data",
  "n_images": 7,
  "board_width": 7,
  "board_height": 5,
  "symmetric": false,
  "r": 25,
  "search_radius": 150,
  "img_ext": "png"
}
```

### 输出文件

| 步骤    | 输出目录              | 文件                                                                 |
| ----- | ----------------- | ------------------------------------------------------------------ |
| 圆心检测  | `centers_out/`    | `centers1.csv` ... `centersN.csv`                                  |
| DIC匹配 | `centersnew_out/` | `centersnew1.csv` ... `centersnewN.csv`                            |
| 双目标定  | `calib_out/`      | `mtxl.csv`, `mtxr.csv`, `distl.csv`, `distr.csv`, `R.csv`, `T.csv` |
| 三维重建  | `recon_out/`      | `pc1.csv`                                                          |

### 代码结构

```
SLCalibration-python/
├── pipeline.py              # 主流程脚本 (命令行入口)
├── dic_matcher.py           # DIC 匹配核心
│   ├── gradient_1d/2d()     #   梯度计算
│   ├── calc_correlation_coefficient()  #   ZNSSD 相关系数
│   ├── interpolation_*()    #   bicubic 样条亚像素插值
│   ├── middle_mat()         #   IC-GN 中间矩阵
│   ├── ic_gn2()             #   IC-GN 单步迭代
│   └── match_one_point() / match_all_points()  #   完整匹配
├── circle_detector.py       # 圆心检测 (OpenCV findCirclesGrid)
├── calibrator.py            # 双目标定 (OpenCV stereoCalibrate)
├── reconstructor.py         # 三维重建 (去畸变 + 最小二乘三角化)
├── utils.py                 # 工具函数
└── requirements.txt         # 依赖
```

### 算法细节

**DIC 匹配** (核心):

1. **整像素搜索**: 以每个圆心为子区中心，在散斑图的 search_radius 范围内用 ZNSSD 相关系数做暴力搜索，找到最近似的整像素位置。

2. **IC-GN 亚像素优化**: 以一阶形函数 (6参数: U, Ux, Uy, V, Vx, Vy) 为变形模型，用 Inverse Compositional Gauss-Newton 方法迭代优化到亚像素精度。收敛条件: ||ΔP|| < 0.01。

3. **bicubic 样条插值**: IC-GN 每次迭代中用 6x6 邻域的 bicubic 样条插值获取亚像素位置的灰度值。

**双目标定**: 先用相机图像标定左相机内参，再用 DIC 匹配到散斑图的点标定"右相机"(投影仪) 内参，最后固定内参做双目立体标定得到外参 R, T。

**三维重建**: 去畸变 (50次迭代) → 线性最小二乘三角化 → 点云居中归一化。

### 引用

如果本代码对您的研究有帮助，请引用:

```
Zhu, C., Zhang, Q., Hou, J., & Gao, H. (2022).
Calibration method for structured light system based on DIC.
Applied Optics, 61(27), 8050-8059.
```

---

## English

### Overview

This repository provides a complete Python implementation of the structured light system calibration method described in **"Calibration Method for Structured Light System Based on DIC"** (Applied Optics, 2022).

The method uses a speckle projector and a single camera, achieving **coding-free** structured light calibration through Digital Image Correlation (DIC) matching — no Gray code or phase shifting patterns required.

### Pipeline

```
Board images → [Circle Detection] → camera-side circle centers
                                        ↓
 Camera img + speckle ref → [DIC Match] → projector-side correspondences
            ↓
 centers(left) + matches(right) + world coords → [Stereo Calib] → params
                                                   ↓
                                     Given matches → [Reconstruct] → point cloud
```

**Key innovation**: Using IC-GN sub-pixel DIC to precisely match calibration circle centers onto the speckle image, treating the projector as a virtual second camera for stereo calibration.

### Requirements

- Python 3.9+
- NumPy, OpenCV, SciPy, Matplotlib

```bash
pip install -r requirements.txt
```

### Data Preparation

Organize your data as follows:

```
your_data_dir/
├── board_pics/              # Calibration board images (camera view)
│   ├── 1.bmp                # Pose 1 (with speckle)
│   ├── 1off.bmp             # Pose 1 (without speckle, optional)
│   ├── 2.bmp
│   └── ...
├── speckle_pic/             # Speckle reference image
│   └── speckle.bmp
├── referencecenters.csv     # World coordinates of circle centers (Nx3)
├── cl.csv                   # Left matching points for reconstruction
└── cr.csv                   # Right matching points for reconstruction
```

Supported image formats: `.bmp`, `.png`, `.jpg`, `.tif`

### Quick Start

```bash
# Full pipeline
python pipeline.py --data_dir ./your_data_dir

# Step by step
python pipeline.py --data_dir ./your_data_dir --mode detect
python pipeline.py --data_dir ./your_data_dir --mode match
python pipeline.py --data_dir ./your_data_dir --mode calib
python pipeline.py --data_dir ./your_data_dir --mode recon
```

### Command-line Arguments

| Argument          | Default | Description                            |
| ----------------- | ------- | -------------------------------------- |
| `--data_dir`      | `.`     | Root data directory                    |
| `--mode`          | `all`   | Run mode: all/detect/match/calib/recon |
| `--n_images`      | `5`     | Number of calibration images           |
| `--board_width`   | `5`     | Circle grid columns                    |
| `--board_height`  | `3`     | Circle grid rows                       |
| `--symmetric`     | `False` | Use symmetric circle pattern           |
| `--r`             | `30`    | DIC subset radius (pixels)             |
| `--search_radius` | `200`   | DIC integer search radius (pixels)     |
| `--config`        | —       | JSON config file path                  |
| `--img_ext`       | `bmp`   | Image file extension                   |

### Output Files

| Step               | Output Dir        | Files                                                              |
| ------------------ | ----------------- | ------------------------------------------------------------------ |
| Circle Detection   | `centers_out/`    | `centers1.csv` ... `centersN.csv`                                  |
| DIC Matching       | `centersnew_out/` | `centersnew1.csv` ... `centersnewN.csv`                            |
| Stereo Calibration | `calib_out/`      | `mtxl.csv`, `mtxr.csv`, `distl.csv`, `distr.csv`, `R.csv`, `T.csv` |
| Reconstruction     | `recon_out/`      | `pc1.csv`                                                          |

### Code Structure

```
SLCalibration-python/
├── pipeline.py              # Main pipeline (CLI entry point)
├── dic_matcher.py           # DIC matching core
├── circle_detector.py       # Circle detection (OpenCV)
├── calibrator.py            # Stereo calibration (OpenCV)
├── reconstructor.py         # 3D reconstruction
├── utils.py                 # Utilities
└── requirements.txt         # Dependencies
```

### Algorithm

**DIC Matching** (core):

1. **Integer-pixel search**: For each circle center, extract a (2r+1)×(2r+1) subset and perform ZNSSD-based brute-force search within search_radius.

2. **IC-GN sub-pixel refinement**: Use first-order shape function (6 parameters) with Inverse Compositional Gauss-Newton optimization. Convergence: ||ΔP|| < 0.01.

3. **Bicubic spline interpolation**: 6×6 neighborhood bicubic spline for sub-pixel gray values during IC-GN iterations.

**Stereo Calibration**: Calibrate left camera intrinsics → calibrate right "camera" (projector) intrinsics using DIC-matched points → stereo calibration with fixed intrinsics.

**3D Reconstruction**: Undistort (50 iterations) → linear least-squares triangulation → point cloud centering.

### Citation

If this code helps your research, please cite:

```
Zhu, C., Zhang, Q., Hou, J., & Gao, H. (2022).
Calibration method for structured light system based on DIC.
Applied Optics, 61(27), 8050-8059.
```
