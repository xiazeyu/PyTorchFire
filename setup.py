from setuptools import setup, find_packages

setup(
    name="pytorchfire",
    version="0.0.1-alpha",
    author="Zeyu Xia",
    author_email="zeyu.xia@email.virginia.edu",
    description="A placeholder package for PyTorchFire",
    long_description="This is a placeholder package for PyTorchFire.",
    long_description_content_type="text/markdown",
    url="https://github.com/xiazeyu/pytorchfire",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
