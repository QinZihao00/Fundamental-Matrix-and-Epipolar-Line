# Compute Fundamental Matrix and Find Epipolar Line
Compute Fundamental Matrix and Epipolar lines using Epipolar Geometry. Just a script for a quick check of your line equation.

1. Note the **coordination system**. 
Convert all your point coordinates into **(b)** as shown in the following image before using the script. The origin is at the center of the image.

![Image of coordinate system](https://github.com/QinZihao00/Fundamental-Matrix-and-Epipolar-Line/blob/master/Coordinate%20System.jpg)


2. Make sure you have `python, numpy and matplotlib` installed. 

3. Add your own point coordinates and modify the following part:
```python
if __name__ == "__main__":
  # Give your eight pairs of points to solve for F
  F = get_FundamentalMatrix(...)
  # Choose a point to get epipolar line equation parameters
  w, b = get_epipolar_param(...)
  # Plot epipolar line on the corresponding image and save as figure
  plot_epipolar_line(...)
```

4. **Good Luck for your Assignment!** :grimacing:
