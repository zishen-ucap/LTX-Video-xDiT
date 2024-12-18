from setuptools import find_packages, setup
import subprocess


def get_cuda_version():
    try:
        nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        version_line = [line for line in nvcc_version.split("\n") if "release" in line][
            0
        ]
        cuda_version = version_line.split(" ")[-2].replace(",", "")
        return "cu" + cuda_version.replace(".", "")
    except Exception as e:
        return "no_cuda"


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    fp = open("xfuser/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    setup(
        name="xfuser",
        author="xDiT Team",
        author_email="fangjiarui123@gmail.com",
        packages=find_packages(),
        install_requires=[
            # "torch==2.3.1",
            "accelerate>=0.33.0",
            "diffusers@git+https://github.com/huggingface/diffusers",  # NOTE: diffusers>=0.32.0.dev is necessary for CogVideoX and Flux
            "transformers>=4.39.1",
            "sentencepiece>=0.1.99",
            "beautifulsoup4>=4.12.3",
            "distvae",
            "yunchang>=0.3.0",
            "pytest",
            "flask",
            "opencv-python",
            "imageio",
            "imageio-ffmpeg",
            "optimum-quanto"
        ],
        extras_require={
            "flash_attn": [
                "flash_attn>=2.6.3",
            ],
        },
        url="https://github.com/xdit-project/xDiT.",
        description="xDiT: A Scalable Inference Engine for Diffusion Transformers (DiTs) on multi-GPU Clusters",
        long_description=long_description,
        long_description_content_type="text/markdown",
        version=version,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
        python_requires=">=3.10",
    )
