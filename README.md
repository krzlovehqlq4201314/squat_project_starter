# Squat Checker (Visual AI Coach)

A simple **Streamlit** app that uses **YOLOv8-Pose** to analyze squat images/webcam frames:
- Draw skeleton lines and keypoint heatmap
- Judge **SQUAT / NOT SQUAT** and provide tips (knee angle, torso lean, hip depth, knee tracking)
- CPU-friendly demo

## Demo

- Upload Image · Analysis: run pose detection on a single JPG/PNG
- Webcam Stable: take a snapshot via your webcam and analyze

## Installation (Windows Git Bash)

```bash
python -m venv .venv
source .venv/Scripts/activate          # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
Run
bash
复制代码
streamlit run app/main.py
Project Structure
css
复制代码
app/
  main.py
  pages/
    10_Upload_Image_Analysis.py
    21_Webcam_Stable.py
  utils/
    pose.py
assets/
License
MIT
