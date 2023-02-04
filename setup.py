import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="classification_bold",
    version="0.0.1",
    author="Dunn Kopylov",
    author_email="38dunn@gmail.com",
    description="getting characteristics about row, "
                "by which you can judge whether it is bold or not",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9"
)
