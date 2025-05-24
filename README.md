# 🖼️ Mediffuse: Diffusion-Driven Translation for MRI Generation from CT Scan

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/) [![Live Demo on Gradio](https://img.shields.io/badge/Gradio-Demo-brightgreen)](https://d969b2857f0f723529.gradio.live/) [![Finetuned Weights and Training Data](https://img.shields.io/badge/Data&Weights-HuggingFace-orange)](https://huggingface.co/sauravdosi) 

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

### Qualitative:

| Input CT                         | Output MRI (Zero-Shot Stable Diffusion 1.5) | Output MRI (ControlNet)                | Output MRI (InstructPix2Pix)           | Output MRI (Ground Truth)              |
|----------------------------------------|---------------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|
| ![00087_ct.png](img/quiz/00087_ct.png) | ![sd.png](img/quiz/sd.png)                  | ![00087_ctcontour_sample_2.png](img/quiz/00087_ctcontour_sample_2.png) | ![ip2p.png](img/quiz/ip2p.png) | ![00087_mr.png](img/quiz/00087_mr.png) |

*Check out more examples in the `results/` folder or try the live demo above.*

---

### Quantitative:

![results.png](img/results.png)


## 📦 Installation

Ensure you have Python 3.10 and Conda installed.

```bash
conda create -n mediffuse python=3.10 -y  
conda activate mediffuse  
pip install -r requirements.txt
