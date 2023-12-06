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
    
    python script.py ./images

If you want to detect single image, put the image in the same directory as the script.py，and run this command:

    python script.py your/image/name

3: if you want to evaluate count performance，It can calculate precision and time simply use this command:

(压缩文件里只有两张照片，没有具体的标签及其形式，所以我只是简单的计算了程序统计的人脸数与实际人脸数的一个对比，TP,TN,FP,FN等具体值暂未计算)

    python script.py ./images -e True
    

