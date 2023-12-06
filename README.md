# Count_faces
Count faces in pictures!

# Rely on
platform: Ubuntu18.04 or Windows10

python==3.10.5

numpy==1.26.2

opencv-python==4.8.1.78

torch==1.12.1+cpu

torchvision==0.13.1+cpu
# Conda installation instruction
Use conda to configuration, The command as follows:

    conda create --name count_face python==3.10.5

    conda activate count_face

    pip install numpy opencv-python

    pip install torch==1.12.1+cpu torchvision==0.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
# Quick start
1: git clone https://github.com/zzq-beyond/count_faces.git

2: Terminal input：

    cd count_faces
    
    python script.py ./images (You can also use your own image path, directory or single image)

3: if you want to evaluate time performance，The command as follows:

    python script.py ./images -t True
    

