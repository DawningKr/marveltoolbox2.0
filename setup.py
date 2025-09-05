from setuptools import setup, find_packages


def get_requires(file_path):
    with open(file_path) as file:
        content = file.readlines()
    return list(map(lambda x: x.strip(), content))


with open("VERSION", "r") as f:
    version = f.read().strip()
    if version.endswith("dev"):
        version = version[:-3]

install_requires = get_requires("./requirements.txt")


setup(
    name="marveltoolbox",
    version=version,
    packages=find_packages(),
    author="DawningKr, Renjie Xie, Wei Xu",
    maintainer="DawningKr",
    url="https://github.com/DawningKr/marveltoolbox2.0",
    description="A toolbox for DL and communication research.",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Utilities",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Environment :: Console",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ],
    keywords="deep learning, "
    "machine learning, supervised learning, "
    "unsupervised learning, communication, "
    "complex value matrix computation",
    python_requires=">=3.11",
    platforms=["Linux"],
    # data_files=[('',['VERSION'])],
    include_package_data=True,
    install_requires=install_requires,
)
