from setuptools import setup

setup(
    name="kokoro-align",
    version="0.1",
    author="Katsuya Iida",
    author_email="katsuya.iida@gmail.com",
    description="Kokoro-Align is a speech-text aligner.",
    license="MIT",
    url="https://github.com/kaiidams/Kokoro-Align",
    packages=['kokoro_align'],
    long_description="""Kokoro-Align is a PyTorch speech-transcript alignment tool for LibriVox.
It splits audio files in silent positions and find CTC best path to
align transcript texts with the audio files.""",
    install_requires=[
        'torch',
        'torchaudio',
        'beautifulsoup4',
        'fugashi',
        'unidic-lite'
    ])
