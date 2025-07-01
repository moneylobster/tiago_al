## TIAGo Abstraction Library (tiago_al)

Provides a Python interface to the various facilities of the TIAGo robot so you don't have to worry about the ROS parts too much.

Represents poses as spatialmath SE3 objects so that they're easy to work with.

### Dependencies

* various python libaries
* espeak-ng for multi-language TTS (this is separate functionality from the built-in one)

### Usage

```python
import tiago_al

tiago=tiago_al.Tiago()
```

Read through `tiago_al.py` to see what functionality is available. Subclassing the Tiago class to add custom behaviors is recommended
