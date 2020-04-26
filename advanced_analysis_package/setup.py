import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="advanced_analysis_package",
    version="0.0.2",
    author="Mrinal Mahajan",
    author_email="mahajanmrinal2013@gmail.com",
    description="A package for advanced level variabe ananlysis and reduction for pandas dataframe",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mrinal-Mahajan/advanced_analysis_package/archive/v_02.tar.gz",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)