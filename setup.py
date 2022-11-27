from setuptools import setup

setup(
    name="kokoro_align",
    version="0.1",
    author="Katsuya Iida",
    author_email="katsuya.iida@gmail.com",
    description="Kokoro-Align is a speech-text aligner.",
    license="MIT",
    url="https://github.com/kaiidams/Kokoro-Align",
    packages=['voice100'],
    long_description="""Kokoro-Align is a PyTorch speech-transcript alignment tool for LibriVox.
It splits audio files in silent positions and find CTC best path to
align transcript texts with the audio files.""",
    install_requires=[
        'torch',
        'torchaudio'
    ],
    extras_require={
        "text": [
            'mecab-python3',
            'unidic-lite',
            'beautifulsoup4',
        ]
    })
