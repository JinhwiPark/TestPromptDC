<h2 align="center"> Test-Time Prompt Tuning for Zero-Shot Depth Completion </h2> <p align="center"> <a href="https://chanhwijeong.github.io/"><strong>Chanhwi Jeong</strong></a> · <a href="https://inhwanbae.github.io/"><strong>Inhwan Bae</strong></a> · <a href="https://jinhwipark.com/"><strong>Jin-Hwi Park*</strong></a> · <a href="https://scholar.google.com/citations?user=Ei00xroAAAAJ"><strong>Hae-Gon Jeon*</strong></a> <br> <strong>ICCV 2025 Highlight Paper</strong> <br> <sub>* Corresponding Authors</sub> </p> <p align="center"> <a href="https://www.jinhwipark.com/Depth-with-Sensors"><strong><code>🌐 Project Page</code></strong></a> <a href="https://openaccess.thecvf.com/content/ICCV2025/html/Jeong_Test-Time_Prompt_Tuning_for_Zero-Shot_Depth_Completion_ICCV_2025_paper.html"><strong><code>📄 ICCV 2025 Paper</code></strong></a> <a href="https://github.com/JinhwiPark/TestPromptDC"><strong><code>💻 Official Code</code></strong></a> </p>

## Summary:
TestPromptDC introduces a test-time prompt tuning framework for zero-shot depth completion across varying sensors and environments,
allowing foundation models for monocular depth estimation to achieve absolute-scale predictions without ground-truth fine-tuning.

## Dataset Download Instructions

### 1. IBIMS-1

- Visit [IBIMS-1 Website](https://www.asg.ed.tum.de/lmf/ibims1/) to download the dataset.
- After downloading, your folder structure should look like:


### 2. DDAD

- Follow the instructions at [this repository](https://github.com/bartn8/vppdc?tab=readme-ov-file#floppy_disk-datasets) to download the DDAD dataset.
- Once downloaded, place the files according to the following structure:


## Dataset Path Setup

- For DDAD testing, set the `--dataset_path` argument to the location of `ddad_train_val` or the folder containing the `ddad` structure above.
- For IBIMS-1, set the `--dataset_path` argument to the `ibims1` folder.

## Usage

Run the following command, replacing each bracketed item with the appropriate value:

- **--gpu**: Specify the GPU number (e.g., `0` for the first GPU).
- **--mode**: Select either `VP` (visual prompt mode) or `FT` (fine-tuning mode).
- **--dataset**: Choose `ibims` or `ddad`.
- **--dataset_path**: Provide the path to your dataset folder.

Example:
python main.py --gpu 0 --mode VP --dataset ddad --dataset_path /path/to/ddad

bibtex
@InProceedings{Jeong_2025_ICCV,
    author    = {Jeong, Chanhwi and Bae, Inhwan and Park, Jin-Hwi and Jeon, Hae-Gon},
    title     = {Test-Time Prompt Tuning for Zero-Shot Depth Completion},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {9443-9454}
}
