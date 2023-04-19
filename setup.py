from setuptools import setup

setup(
    name='fasdr',
    version='0.1',
    author='David Marx',
    author_email='david.marx84@gmail.com',
    description='A library for dense retrieval using FAISS and Sentence-BERT',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/dmarx/fasdr',
    packages=['fasdr'],
    install_requires=[
        'numpy',
        'spacy',
        'sentence_transformers',
        'scipy',
        #'en_core_web_sm',
        'spacy-sentence-bert',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
