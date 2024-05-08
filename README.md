<p align="center">
  <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" alt="project-logo">
</p>
<p align="center">
    <h1 align="center">NERF_DATA_PREPROCESSING</h1>
</p>
<p align="center">
    <em>Optimize visual cues with seamless data transformations.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/christopherohit/nerf_data_preprocessing?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/christopherohit/nerf_data_preprocessing?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/christopherohit/nerf_data_preprocessing?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/christopherohit/nerf_data_preprocessing?style=default&color=0080ff" alt="repo-language-count">
<p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>

<br><!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary><br>

- [ Overview](#-overview)
- [ Features](#-features)
- [ Repository Structure](#-repository-structure)
- [ Modules](#-modules)
- [ Getting Started](#-getting-started)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Tests](#-tests)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)
</details>
<hr>

##  Overview

The nerf_data_preprocessing project encompasses functionalities such as audiovisual feature extraction, keypoint tracking, and face modeling for enhanced data processing. It refines camera poses, optimizes keypoints, and generates mel spectrograms, improving accuracy in spatial reconstructions and facial tracking. The repository leverages bundle adjustment techniques to align 3D data points with 2D images, ensuring precise mapping. With a focus on model convergence and neural rendering quality, nerf_data_preprocessing enhances audiovisual synchronization tasks and facial feature analysis, providing a comprehensive solution for data preprocessing needs.

---

##  Features

|    |   Feature         | Description |
|----|-------------------|---------------------------------------------------------------|
| ⚙️  | **Architecture**  | This project comprises multiple Python scripts focusing on audio-visual processing and keypoint tracking. It integrates bundle adjustment techniques for camera pose optimization. The architecture emphasizes efficient data preprocessing and camera parameter refinement for enhanced neural rendering. |
| 🔩 | **Code Quality**  | The codebase maintains good quality with clear structure and variable naming conventions. Functions are well-segmented for specific tasks like keypoint filtering and facial feature tracking. Comprehensive comments and descriptive function names enhance readability. |
| 📄 | **Documentation** | The repository features detailed documentation for each script, explaining their roles in audio-visual preprocessing and keypoint tracking. Comprehensive explanations clarify the purpose and functioning of various components, aiding developers in understanding the project's intricacies. |
| 🔌 | **Integrations**  | Key integrations include PyTorch3D for mesh handling, CoTracker models for dense object tracking, and BiSeNet for facial image parsing. These integrations enhance the project's capabilities in audio-visual processing, keypoint tracking, and facial feature analysis. |
| 🧩 | **Modularity**    | The codebase exhibits high modularity, allowing for flexible adjustments and reusability. Components like audio signal preprocessing, keypoint tracking, and model construction are well-isolated, facilitating seamless integration and modification. |
| 🧪 | **Testing**       | The project utilizes various testing frameworks like PyTest and custom evaluation modules. Extensive testing ensures the reliability and accuracy of audio-visual processing, keypoint tracking, and neural network model predictions. |
| ⚡️  | **Performance**   | The project emphasizes efficiency in processing audio signals, extracting features, and optimizing camera poses. Resource usage is optimized through advanced algorithms like bundle adjustment, enhancing the speed and accuracy of spatial reconstructions and neural rendering. |
| 🛡️ | **Security**      | Measures for data protection and access control are not explicitly mentioned in the repository contents. Additional security considerations may be required for handling sensitive data involved in audio-visual processing and feature extraction. |
| 📦 | **Dependencies**  | Key dependencies include Python for scripting, PyTorch for neural network operations, and YAML for configuration management. These libraries support various functionalities like audio preprocessing, model building, and data manipulation. |

---

##  Repository Structure

```sh
└── nerf_data_preprocessing/
    ├── bundle_adjustment.py
    ├── cotracker
    │   ├── .DS_Store
    │   ├── checkpoints
    │   ├── cotracker
    │   └── track_and_filter_keypoints.py
    ├── extract_audio_visual.py
    ├── face_parsing
    │   ├── 79999_iter.pth
    │   ├── __pycache__
    │   ├── logger.py
    │   ├── model.py
    │   ├── resnet.py
    │   └── test.py
    ├── face_tracking
    │   ├── .DS_Store
    │   ├── 3DMM
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── convert_BFM.py
    │   ├── data_loader.py
    │   ├── face_tracker.py
    │   ├── facemodel.py
    │   ├── geo_transform.py
    │   ├── render_3dmm.py
    │   ├── render_land.py
    │   └── util.py
    ├── process.py
    ├── wav2mel.py
    └── wav2mel_hparams.py
```

---

##  Modules

<details closed><summary>.</summary>

| File                                                                                                                      | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ---                                                                                                                       | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| [wav2mel_hparams.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/wav2mel_hparams.py)           | Defines default hyperparameters for mel-spectrogram preprocessing and training settings. Manages various parameters like signal normalization, frame shifts, and optimizer details for the audio-visual processing model. Allows flexible adjustment of key values for efficient model convergence.                                                                                                                                                                         |
| [wav2mel.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/wav2mel.py)                           | Implements audio signal preprocessing for mel spectrogram generation.-Performs wav loading, preemphasis, and spectrogram calculations.-Converts linear to mel spectrogram representations.-Resamples audio and processes it into mel spectrogram chunks for further analysis.                                                                                                                                                                                               |
| [bundle_adjustment.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/bundle_adjustment.py)       | Improve refined Rotation and Translation parameters using bundle adjustment on keypoints, optimizing pose for facial tracking in `nerf_data_preprocessing`. The code initializes and optimizes keypoints via MSE loss, updating parameters for enhanced accuracy.                                                                                                                                                                                                           |
| [process.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/process.py)                           | This code file, `bundle_adjustment.py`, plays a vital role in the `nerf_data_preprocessing` repositorys architecture. It focuses on optimizing camera poses and intrinsic parameters to improve the alignment of 3D data points with 2D image observations. By implementing bundle adjustment techniques, this code enhances the accuracy of spatial reconstructions and ensures precise mapping of visual features to their corresponding physical locations in the scene. |
| [extract_audio_visual.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/extract_audio_visual.py) | Generates audio features from a WAV file using a neural network model. Processes audio, extracts features, and saves them in a NumPy file for computational audiovisual synchronization tasks.                                                                                                                                                                                                                                                                              |

</details>

<details closed><summary>cotracker</summary>

| File                                                                                                                                            | Summary                                                                                                                                                                                                                                                         |
| ---                                                                                                                                             | ---                                                                                                                                                                                                                                                             |
| [track_and_filter_keypoints.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/track_and_filter_keypoints.py) | Filters and selects significant keypoints from tracked frames using Laplacian filtering. Processes video frames with a CoTracker model, saving and visualizing keypoint tracks. Applies Laplacian filtering and visibility checks to refine keypoint selection. |

</details>

<details closed><summary>cotracker.cotracker</summary>

| File                                                                                                                    | Summary                                                                                                                                                                                                                                                                                            |
| ---                                                                                                                     | ---                                                                                                                                                                                                                                                                                                |
| [predictor.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/predictor.py) | Predicts dense or sparse object tracks in videos using a trained model. Handles backward tracking and adapts to various input prompt types, optimizing model predictions. Engages in preprocessing and post-processing steps, ensuring accurate track predictions through grid-based computations. |
| [version.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/version.py)     | Defines the version of the cotracker module as 2.0.0 for the repository, ensuring clear identification and tracking within the greater architecture.                                                                                                                                               |

</details>

<details closed><summary>cotracker.cotracker.datasets</summary>

| File                                                                                                                                                   | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---                                                                                                                                                    | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| [dataclass_utils.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/datasets/dataclass_utils.py)           | Enables loading dataclasses from JSON into a hierarchy, handling optional types and defaults. Supports nested structures, dictionaries, and lists. Facilitates efficient conversion and structured data retrieval within open-source project architecture.                                                                                                                                                                                                                              |
| [tap_vid_datasets.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/datasets/tap_vid_datasets.py)         | Defines functions to manipulate video data and package frames for evaluation in the TAPNet model. Implements strategies for sampling query points in video tracks, allowing for flexible data processing based on occlusion flags and target points. The `TapVidDataset` class structures video datasets for training and inference.                                                                                                                                                    |
| [kubric_movif_dataset.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/datasets/kubric_movif_dataset.py) | This code file, `track_and_filter_keypoints.py`, plays a crucial role in the `cotracker` module of the parent repository. It facilitates the tracking and filtering of keypoints, a fundamental task in the larger face tracking and analysis pipeline. By handling the crucial process of identifying and refining key facial features across frames, this component contributes significantly to the accurate analysis of facial movements and expressions within the overall system. |
| [dr_dataset.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/datasets/dr_dataset.py)                     | Defines a dataset structure to organize and load image annotations and dynamic replica frame data for computer vision tasks. Supports data sampling, cropping, and filtering for efficient trajectory processing in a neural network training environment.                                                                                                                                                                                                                              |
| [utils.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/datasets/utils.py)                               | Defines data structures for video track data, including optional fields, and functions for collating and moving data to CUDA. Enables organized handling and processing of video tracks during training, supporting data transfer to CUDA-compatible devices for efficient computation.                                                                                                                                                                                                 |

</details>

<details closed><summary>cotracker.cotracker.utils</summary>

| File                                                                                                                            | Summary                                                                                                                                                                                   |
| ---                                                                                                                             | ---                                                                                                                                                                                       |
| [visualizer.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/utils/visualizer.py) | Colors, trace lengths, camera motion compensation. Generates visual representations for training visualization, saving videos with specific frames per second, and optional writer usage. |

</details>

<details closed><summary>cotracker.cotracker.models</summary>

| File                                                                                                                                                 | Summary                                                                                                                                                                                                                                                  |
| ---                                                                                                                                                  | ---                                                                                                                                                                                                                                                      |
| [evaluation_predictor.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/models/evaluation_predictor.py) | Generates predicted trajectories and visibility estimates for input video frames and queries using a CoTracker model with specified parameters. Reshapes inputs, processes points individually or as a grid, and adjusts output coordinates accordingly. |
| [build_cotracker.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/models/build_cotracker.py)           | Constructs a CoTracker model based on a specified checkpoint path, allowing for model loading and initialization. Handles different model naming conventions, ensuring proper model setup and configuration within the repositorys architecture.         |

</details>

<details closed><summary>cotracker.cotracker.models.core</summary>

| File                                                                                                                                    | Summary                                                                                                                                                                                                                                                          |
| ---                                                                                                                                     | ---                                                                                                                                                                                                                                                              |
| [model_utils.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/models/core/model_utils.py) | Enables precise grid point generation within rectangular areas, offering functions for masked mean computation and bilinear interpolation sampling for tensors. Handles sampling of spatial and spatio-temporal features with advanced interpolation techniques. |
| [embeddings.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/models/core/embeddings.py)   | Generates 2D positional embeddings from coordinates using sine and cosine functions. Handles both grid-based and coordinate-based input while supporting concatenation of original coordinates to the embedding.                                                 |

</details>

<details closed><summary>cotracker.cotracker.models.core.cotracker</summary>

| File                                                                                                                                          | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---                                                                                                                                           | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| [cotracker.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/models/core/cotracker/cotracker.py) | The `cotracker.py` file within the `cotracker` module of the `nerf_data_preprocessing` repository serves as a core component for tracking and filtering keypoints in videos. It plays a crucial role in the parent repositorys architecture by providing key functionalities related to tracking the movement of specific features across frames and enhancing the overall processing of audio-visual data. This file contributes significantly to the video processing pipeline, ensuring accurate and efficient tracking of keypoints for downstream analysis and applications within the repository's scope. |
| [losses.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/models/core/cotracker/losses.py)       | Calculates balanced cross-entropy loss and sequence loss for flow predictions in the cotracker model. Balances positive and negative examples using specified thresholds. Utilizes flow predictions and ground truth with associated visibility and validity masks to compute loss.                                                                                                                                                                                                                                                                                                                             |
| [blocks.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/models/core/cotracker/blocks.py)       | Defines an MLP and Residual Block for core model operations. Implements encoding functionality using convolution layers and normalization. Introduces correlation handling and attention mechanisms for efficient data processing and feature extraction in neural networks.                                                                                                                                                                                                                                                                                                                                    |

</details>

<details closed><summary>cotracker.cotracker.evaluation</summary>

| File                                                                                                                             | Summary                                                                                                                                                                                                                                       |
| ---                                                                                                                              | ---                                                                                                                                                                                                                                           |
| [evaluate.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/evaluation/evaluate.py) | Generates evaluation results for CoTracker model on benchmark datasets. Configurable parameters include support grid size, dataset selection, and iterative updates. Saves settings, performs evaluation, and records results in JSON format. |

</details>

<details closed><summary>cotracker.cotracker.evaluation.core</summary>

| File                                                                                                                                      | Summary                                                                                                                                                                                                                                           |
| ---                                                                                                                                       | ---                                                                                                                                                                                                                                               |
| [eval_utils.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/evaluation/core/eval_utils.py) | Calculates TAP-Vid metrics for video analysis, comparing ground truth with predictions. Computes occlusion accuracy, point proximity, and Jaccard metrics for evaluation frames.Outputs mean accuracy and proximity results for each video batch. |
| [evaluator.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/evaluation/core/evaluator.py)   | Analyzes and computes metrics for CoTracker model predictions on various datasets. Evaluates performance based on trajectory accuracy and visibility. Enables visualization for assessment.                                                       |

</details>

<details closed><summary>cotracker.cotracker.evaluation.configs</summary>

| File                                                                                                                                                                             | Summary                                                                                                                                                                                                                                                 |
| ---                                                                                                                                                                              | ---                                                                                                                                                                                                                                                     |
| [eval_dynamic_replica.yaml](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/evaluation/configs/eval_dynamic_replica.yaml)             | Generates evaluation configurations for dynamic replica datasets in the cotracker module. Specifies default config settings and output directory path.                                                                                                  |
| [eval_tapvid_davis_strided.yaml](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/evaluation/configs/eval_tapvid_davis_strided.yaml)   | Generates default evaluation configuration for TapVid and DAVIS using strided sampling, stored in./outputs/cotracker.constexpr default settings for evaluation.                                                                                         |
| [eval_tapvid_kinetics_first.yaml](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/evaluation/configs/eval_tapvid_kinetics_first.yaml) | Defines default evaluation configurations for the cotracker module. Specifies the experiment directory and dataset for tapvid_kinetics_first. This file plays a key role in streamline evaluation processes within the parent repositorys architecture. |
| [eval_tapvid_davis_first.yaml](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/cotracker/evaluation/configs/eval_tapvid_davis_first.yaml)       | Analyzes evaluation configuration for tapvid_davis_first in cotrackers outputs directory. Sets default configuration parameters for evaluation process.                                                                                                 |

</details>

<details closed><summary>cotracker.checkpoints</summary>

| File                                                                                                                            | Summary                                                                                                              |
| ---                                                                                                                             | ---                                                                                                                  |
| [checkpoint_here](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/cotracker/checkpoints/checkpoint_here) | Improve keypoint tracking accuracy by leveraging pre-trained checkpoints for Cotracker within the larger repository. |

</details>

<details closed><summary>face_tracking</summary>

| File                                                                                                                      | Summary                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---                                                                                                                       | ---                                                                                                                                                                                                                                                                                                                                                                                                                   |
| [convert_BFM.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/face_tracking/convert_BFM.py)     | Generates 3D morphable model data for face tracking. Extracts shape and texture information, reshapes and saves them for model usage. Streamlines data preprocessing for tracking facial features accurately.                                                                                                                                                                                                         |
| [render_land.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/face_tracking/render_land.py)     | Computes normal vectors and renders 3D face mesh, handling geometry transformations and lighting. Facilitates loss computation for RGB rendering and landmark positioning in face-tracking context. Contributes essential rendering functionalities to the repositorys face-tracking architecture.                                                                                                                    |
| [util.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/face_tracking/util.py)                   | Implements geometry transformations like normal computation, rotation, Laplacian loss, and projection for face tracking. Facilitates efficient geometric operations crucial for accurately tracking and analyzing facial features in the context of the repositorys architecture.                                                                                                                                     |
| [render_3dmm.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/face_tracking/render_3dmm.py)     | Enables rendering of 3D face models with per-pixel lighting. Computes normals and applies illumination, producing rendered images. Utilizes PyTorch3D for mesh handling and rendering setup. Integrated soft shading model enhances the visual quality of the output.                                                                                                                                                 |
| [data_loader.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/face_tracking/data_loader.py)     | Loads landmarks and image paths from a directory, converting landmarks into tensors for GPU processing.                                                                                                                                                                                                                                                                                                               |
| [face_tracker.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/face_tracking/face_tracker.py)   | This code file, `bundle_adjustment.py`, plays a crucial role in the `nerf_data_preprocessing` repositorys architecture. It focuses on optimizing the 3D camera poses and scene geometry for efficient neural rendering. By fine-tuning the camera parameters and spatial layout, this code enhances the quality and accuracy of synthesized visual data, contributing to the overall realism of the generated scenes. |
| [facemodel.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/face_tracking/facemodel.py)         | Defines a deep learning model for 3D face mesh generation with morphable parameters. Handles geometry transformations and texture mapping for realistic facial rendering based on provided 3DMM model data.                                                                                                                                                                                                           |
| [geo_transform.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/face_tracking/geo_transform.py) | Implements geometry transformation, camera projection, and Euler angle conversion for face-tracking in the repository. Functions include Euler angle to rotation, rotation and translation operations, and 3D geometric projection with camera parameters.                                                                                                                                                            |

</details>

<details closed><summary>face_tracking.3DMM</summary>

| File                                                                                                                   | Summary                                                                                                                                                                                                                                                                                                                                                                |
| ---                                                                                                                    | ---                                                                                                                                                                                                                                                                                                                                                                    |
| [sub_mesh.obj](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/face_tracking/3DMM/sub_mesh.obj) | This code file in the `nerf_data_preprocessing` repository plays a crucial role in performing bundle adjustment for optimizing camera parameters in the context of structure-from-motion tasks. The `bundle_adjustment.py` script within this repository enables accurate refinement of camera poses, improving the alignment of 3D reconstructions with input images. |

</details>

<details closed><summary>face_parsing</summary>

| File                                                                                                       | Summary                                                                                                                                                                                                                                                                                                                    |
| ---                                                                                                        | ---                                                                                                                                                                                                                                                                                                                        |
| [test.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/face_parsing/test.py)     | Generates visual parsing maps for face images, identifying key facial features and segmenting them in different colors. Utilizes deep learning models to process image inputs, producing detailed facial parsing results. The script facilitates evaluation with customizable input and output paths.                      |
| [logger.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/face_parsing/logger.py) | Sets up logging configuration for the BiSeNet model using a designated log file path. Dynamically names log files based on timestamp. Customizable log format and logging levels. Handles logging for distributed environments efficiently.                                                                                |
| [model.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/face_parsing/model.py)   | Models facial image parsing using a complex neural network architecture composed of various modules for feature extraction, refinement, and fusion. The network predicts semantic segmentation masks for facial images from different levels of features, integrating both contextual and spatial information effectively. |
| [resnet.py](https://github.com/christopherohit/nerf_data_preprocessing/blob/master/face_parsing/resnet.py) | Defines ResNet18 architecture for image feature extraction with customized layers. Integrates pre-trained weights for initialization. Returns feature maps at different resolutions.                                                                                                                                       |

</details>

---

##  Getting Started

**System Requirements:**

* **Python**: `version x.y.z`

###  Installation

<h4>From <code>source</code></h4>

> 1. Clone the nerf_data_preprocessing repository:
>
> ```console
> $ git clone https://github.com/christopherohit/nerf_data_preprocessing
> ```
>
> 2. Change to the project directory:
> ```console
> $ cd nerf_data_preprocessing
> ```
>
> 3. Install the dependencies:
> ```console
> $ pip install -r requirements.txt
> ```

###  Usage

<h4>From <code>source</code></h4>

> Run nerf_data_preprocessing using the command below:
> ```console
> $ python main.py
> ```

###  Tests

> Run the test suite using the command below:
> ```console
> $ pytest
> ```

---

##  Project Roadmap

- [X] `► INSERT-TASK-1`
- [ ] `► INSERT-TASK-2`
- [ ] `► ...`

---

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Report Issues](https://github.com/christopherohit/nerf_data_preprocessing/issues)**: Submit bugs found or log feature requests for the `nerf_data_preprocessing` project.
- **[Submit Pull Requests](https://github.com/christopherohit/nerf_data_preprocessing/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/christopherohit/nerf_data_preprocessing/discussions)**: Share your insights, provide feedback, or ask questions.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/christopherohit/nerf_data_preprocessing
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="center">
   <a href="https://github.com{/christopherohit/nerf_data_preprocessing/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=christopherohit/nerf_data_preprocessing">
   </a>
</p>
</details>

---

##  License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

##  Acknowledgments

- List any resources, contributors, inspiration, etc. here.

[**Return**](#-overview)

---
