# ChestX-GAN: Intelligent Chest X-ray Enhancement & Multiclass Detection  

An AI-powered system for enhancing chest X-ray images and detecting multiple thoracic diseases in real time. The project integrates image enhancement using GANs, object detection with YOLO, medical text generation with NLP, and audio output using gTTS to improve accessibility.

---

## Key Features  

- Enhances low-resolution chest X-ray images using **GANs** to improve clarity for diagnosis  
- Detects multiple lung conditions (e.g., pneumonia, tuberculosis, lung opacity) using **YOLO**  
- Generates patient-friendly diagnostic explanations using **NLP (GPT-2)**  
- Converts medical summaries to natural speech using **Google Text-to-Speech (gTTS)**  
- Provides interactive Q&A on diagnosis using NLP  

---

## Components  

- `gan_enhancement.ipynb`: GAN model for X-ray image enhancement  
- `yolo_detection.ipynb`: YOLO-based multiclass detection of chest X-ray abnormalities  
- `nlp_tts.ipynb`: NLP-based explanation generation and speech synthesis  
- `requirements.txt`: List of required Python libraries  

---

## Libraries Used  

- `torch`, `torchvision`, `torchxrayvision`  
- `transformers` (Hugging Face)  
- `ultralytics` (YOLO)  
- `gTTS`, `IPython.display`  
- `Pandas`, `NumPy`, `Matplotlib`, `OpenCV`, `PIL`  

---

## How It Works  

1. Enhances chest X-ray images with GAN to reduce noise and improve resolution  
2. Runs YOLO for real-time multiclass detection of lung abnormalities  
3. Uses NLP (GPT-2) to generate medical explanations based on detected conditions  
4. Converts text explanations into speech using gTTS  
5. Provides a Q&A interface for user questions about the diagnosis  

---

## Usage  

- Run each notebook (`gan_enhancement.ipynb`, `yolo_detection.ipynb`, `nlp_tts.ipynb`) in order  
- Ensure all dependencies from `requirements.txt` are installed  
- Interact via notebook cells to generate explanations and speech outputs  

---

## Example  

```bash
# Example (YOLO training command)
model.train(data="data.yaml", epochs=40, imgsz=512)
