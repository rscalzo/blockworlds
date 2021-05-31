import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('LICENSE') as f:
    license = f.read()


setuptools.setup(
    name="blockworlds",
    version="0.1.0",
    author="Richard Scalzo",
    author_email="richard.scalzo@sydney.edu.au",
    description="Anti-aliased kinematic models for structural geology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=('notebooks','docs',"scripts")),
    license=license,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
