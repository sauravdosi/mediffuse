# 🖼️ Mediffuse: CT ↔️ MRI Diffusion-Driven Cross-Modal Translation

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

### Comparative:
![comparison.png](img/results/comparison.png)

### CT → MRI (ControlNet):

![results_1.png](img/results/results_1.png)

### CT → MRI (InstructPix2Pix):

![results_2.png](img/results/results_2.png)

### MRI → CT (InstructPix2Pix):

![mri2ct.png](img/results/mri2ct.png)

*Check out more examples in the `results/` folder or try the live demo above.*

---

### Quantitative:

![results.png](img/results/results.png)


### Evaluation:




## 📦 Installation

Ensure you have Python 3.10 and Conda installed.

```bash
conda create -n mediffuse python=3.10 -y  
conda activate mediffuse  
pip install -r requirements.txt
