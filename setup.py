import setuptools

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'Pillow', 'scikit-image', 'opencv-python', 'tqdm', 'imageio']

setuptools.setup(
    name='human_inst_seg',
    url='https://github.com/Project-Splinter/human_inst_seg',
    description='A Single Human Instance Segmentor runs at 50 FPS on GV100', 
    version='0.0.2',
    author='Ruilong Li',
    author_email='ruilongl@usc.edu',    
    license='MIT License',
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIREMENTS + [
        'segmentation_models_pytorch@git+https://github.com/qubvel/segmentation_models.pytorch',
        'human_det@git+https://github.com/Project-Splinter/human_det',
    ]
)

