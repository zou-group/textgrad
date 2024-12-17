import os
import subprocess

from setuptools import find_packages, setup
from setuptools.command.install import install


with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()


class CustomInstallCommand(install):
    """Customized setuptools install command - compiles .po files to .mo."""

    def run(self):
        # Compile .po files to .mo files before proceeding with installation
        self.compile_translations()
        # Run the standard install process
        install.run(self)

    def compile_translations(self):
        locales_dir = os.path.join('textgrad', 'locales')
        for lang in os.listdir(locales_dir):
            po_file = os.path.join(locales_dir, lang, 'LC_MESSAGES', 'textgrad.po')
            mo_file = os.path.join(locales_dir, lang, 'LC_MESSAGES', 'textgrad.mo')
            if os.path.exists(po_file):
                print(f'Compiling {po_file} to {mo_file}')
                try:
                    subprocess.run(['msgfmt', po_file, '-o', mo_file], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Warning: Failed to compile {po_file}. Installation will continue.\nError: {e}")
            else:
                print(f'No .po file found for language {lang}')

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
    cmdclass={
        'install': CustomInstallCommand,
    },
    include_package_data=True,
    package_data={"textgrad": ["locales/*/LC_MESSAGES/*.mo"]},
    install_requires=requirements,
    extras_require={
        "vllm": ["vllm"],
    },
    zip_safe=False,
)
