import setuptools

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'Pillow', 'opencv-python', 'tqdm', 'imageio']

setuptools.setup(
    name='human_det',
    url='https://github.com/liruilong940607/human_det',
    description='A Single Human Detector runs at 70 FPS on GV100',    
    version='0.0.1',
    author='Ruilong Li',
    author_email='ruilongl@usc.edu',    
    license='MIT License',
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIREMENTS,
)

