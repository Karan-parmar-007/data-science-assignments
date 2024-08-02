from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT= '-e .'
def get_requirements(file_path:str)->List[str]:
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

        return requirements

setup(
    name = 'BreastCancerPrediction',
    version = '0.0.1',
    author = 'Karan',
    author_email = 'karanparmar00014@gmail.com',
    install_requires =  get_requirements('requirements.txt'),
    packages = find_packages(),
    description = 'Predicting if you have breast cancer',
    long_description = 'This is a small project, no one knows if it is correct 🤣🤣',
)