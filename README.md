# gan-toy
--------------- 

a small toy for tracking image generation from a bunch of sample images.

1. Running with main.py
----------------------------

`pip install -r requirements.txt`
`python main.py`

2. Building an Executable with PyInstaller
--------------------------------------------

`pip install pyinstaller`
`pyinstaller --onefile --noconsole main.py`

```
dist/
├── main.exe   <- Your standalone executable
```