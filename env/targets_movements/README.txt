# Pathfinding and Bezier Curve Plotting

This project includes scripts to perform pathfinding around obstacles, generate Bezier curves, and plot the resulting path.

## Files

1. **load_obstacles.py**
   - Description: Loads obstacles from a JSON file.
   - Usage: `python load_obstacles.py`

2. **bezier.py**
   - Description: Contains functions to generate a Bezier curve.
   - Usage: `python bezier.py`

3. **pathfinding.py**
   - Description: Implements pathfinding from a start point to an end point, avoiding obstacles.
   - Usage: `python pathfinding.py`

4. **utils.py**
   - Description: Utility functions used in the project, such as checking if a point is within an obstacle and generating points around obstacles.

5. **plot_path.py**
   - Description: Plots the obstacles, control points, and the Bezier curve representing the path.
   - Usage: `python plot_path.py`

## Workflow

1. Load obstacles from the JSON file using `load_obstacles.py`.
2. Use `pathfinding.py` to generate a path from the start point to the end point avoiding obstacles.
3. Generate a Bezier curve from the path control points using `bezier.py`.
4. Plot the path and obstacles using `plot_path.py`.
5. The trajectory is saved in `target_policies/trajectory_toy.txt`.

## Dependencies

- `numpy`
- `scipy`
- `matplotlib`
