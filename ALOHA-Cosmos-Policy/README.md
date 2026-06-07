---
license: cc
---
# ALOHA-Cosmos-Policy

## Dataset Description

ALOHA-Cosmos-Policy is a real-world bimanual manipulation dataset collected on the ALOHA 2 robot platform as part of the Cosmos Policy project. This is the dataset used to train the [Cosmos-Policy-ALOHA-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-ALOHA-Predict2-2B) checkpoint.

### Dataset Characteristics

- **Robot platform**: ALOHA 2 (bimanual setup with two ViperX 300 S robot arms)
- **Data type**: Real-world human-teleoperated demonstrations
- **Control frequency**: 25 Hz (reduced from the original 50 Hz to save disk space and increase training speed while maintaining smoothness)
- **Camera views**: 3 (1 top-down + 2 wrist-mounted)
- **Total demonstrations**: 185 successful demonstrations across 4 tasks
- **Data format**: HDF5 files with MP4 video compression for image observations
- **Image resolution**: 256×256 pixels (resized from the original 480×640 raw images)

### Preprocessing

This dataset has been preprocessed from the raw ALOHA teleoperation data with the following modifications:

1. **Image resizing**: Camera images resized from 480×640 to 256×256 pixels
2. **Video compression**: Image sequences converted to MP4 videos (25 fps) for efficient storage
3. **Relative actions**: Computed and stored alongside absolute actions for policy training flexibility (though only absolute actions are used in the Cosmos Policy paper)

### Tasks and Demonstrations

| Task | # Demos | Description |
|------|---------|-------------|
| put X on plate | 80 | Place objects (purple eggplant or brown chicken wing) on a plate based on language instructions |
| fold shirt | 15 | Fold one of three T-shirts in multiple steps, testing long-horizon contact-rich manipulation |
| put candies in bowl | 45 | Collect scattered candies, testing ability to handle multimodal grasp sequences |
| put candy in ziploc bag | 45 | Open and place items in a ziploc slider bag, testing high-precision manipulation with millimeter tolerance |

### Data Format

Each episode HDF5 file contains:
```
# Datasets (arrays)
/observations/qpos          # Joint positions, shape: (T, 14)
/observations/qvel          # Joint velocities, shape: (T, 14)
/observations/effort        # Joint efforts/torques, shape: (T, 14)
/observations/video_paths/  # Video file paths (strings)
    cam_high               # Relative path to top-down camera MP4
    cam_left_wrist         # Relative path to left wrist camera MP4
    cam_right_wrist        # Relative path to right wrist camera MP4
/action                     # Absolute action sequence, shape: (T, 14)
/relative_action            # Relative action sequence (frame-to-frame deltas), shape: (T, 14)
```

### Statistics

- **Total demonstrations**: 185
- **Success rate**: 100% (only successful demonstrations included)
- **Image resolution**: 256×256×3 (RGB, resized from 480×640)
- **Action dimensions**: 14 (7 per arm: joint positions)
- **Proprioception dimensions**: 14 (7 joint angles per arm)
- **Control frequency**: 25 Hz
- **Video FPS**: 25 fps

### ALOHA Robot Platform

This dataset was collected using a robot setup similar to the ALOHA 2 system:
- **Paper**: [ALOHA 2: An Enhanced Low-Cost Hardware for Bimanual Teleoperation](https://aloha-2.github.io/)
- **Repository**: https://github.com/tonyzhaozh/aloha/tree/main/aloha2
- **Hardware**: Two ViperX 300 S robot arms with three cameras
- **License**: MIT License

### Citation

If you use this dataset, please cite the Cosmos Policy paper by Kim et al.
<!-- ```bibtex
# TODO: Add Cosmos Policy BibTeX
``` -->

### License

Creative Commons Attribution 4.0 International (CC BY 4.0)