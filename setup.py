import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='cpgan',
    version='0.0.1',
    author='Zihan Ren',
    author_email='zur74@psu.edu',
    description='Conditional Generation of Porous media image using CGAN',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mike-huls/toolbox',
    project_urls = {
        "Bug Tracker": "https://github.com/mike-huls/toolbox/issues"
    },
    license='MIT'
)