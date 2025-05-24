# 🖼️ Mediffuse: Diffusion-Driven Translation for MRI Generation from CT Scan

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/) [![Live Demo on Gradio](https://img.shields.io/badge/Gradio-Demo-brightgreen)](GRADIO_DEMO_LINK) [![Finetuned Weights](https://img.shields.io/badge/Weights-Finetuned-blue)](FINETUNED_WEIGHTS_LINK) [![Training Data](https://img.shields.io/badge/Data-Training-lightgrey)](TRAIN_DATA_LINK) [![Dataset on Hugging Face](https://img.shields.io/badge/Dataset-HuggingFace-orange)](HUGGINGFACE_DATASET_LINK)

---

## 🚀 Workflow

![Mediffuse Workflow](img/mediffuse.gif)

A quick overview of the end-to-end pipeline:

1. **Data Preparation:** Load paired CT–MRI scans, preprocess and tokenize.  
2. **Model Training:** Fine-tune diffusion models (InstructPix2Pix & ControlNet).  
3. **Inference:** Generate CT→MRI or MRI→CT translations via scripts or interactive Gradio.  
4. **Evaluation:** Quantitative (PSNR, SSIM) and qualitative (visual samples).

---

## 🚀 Features

- 🔄 **Bidirectional Conversion:** CT→MRI & MRI→CT on demand  
- 🤖 **ControlNet Extension:** Conditioning on segmentation masks or edge maps  
- ⚡ **Fast Inference:** Optimized for GPU acceleration  
- 🎨 **Interactive Demo:** Tweak prompts, adjust strength, visualize in real time  

---

## 🖼️ Results

| Input (CT)                        | Output (MRI)                      | Input (MRI)                       | Output (CT)                       |
| --------------------------------- | --------------------------------- | --------------------------------- | --------------------------------- |
| ![](path/to/ct_sample1.png)       | ![](path/to/mri_out1.png)         | ![](path/to/mri_sample2.png)      | ![](path/to/ct_out2.png)          |

*Check out more examples in the `results/` folder or try the live demo above.*

---

## 📦 Installation

Ensure you have Python 3.10 and Conda installed.

```bash
conda create -n mediffuse python=3.10 -y  
conda activate mediffuse  
pip install -r requirements.txt
