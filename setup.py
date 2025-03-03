from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="textgrad",
    version="0.1.6",
    description="",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
    ],
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/zou-group/textgrad",
    author="Zou Group",
    author_email="merty@stanford.edu",
    packages=find_packages(include=["textgrad", "textgrad.*"]),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "vllm": ["vllm"],
    },
    zip_safe=False,
)

