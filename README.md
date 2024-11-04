# Note/idea on this ocntribution
Enhance the viewer
  x Discuss format
    x Idea leave viewing to existing program
      - stl, ply and vrml
        > Stl
          - no color, less good in long term
        > ply
          - As color, seems starndad
          - Created with trimesh create mesh and create ply file
  x Better viewer
    ~ add direction of detectors
    x view fov
  - Clean up a little

  - Extract lut and save it
  - displaying normalisation
    - Show relative effeciency, alpha or color?
  - Display doi ou material
  - Crystal intersections, debug feature
  - Add simplified view


# PETSIRD use case: Scanner viewer extended

The purpose of this repo is to provide capabilities to reproducea scanner geometry using either the STL or PLY 3D format.

## Background
The [Emission Tomography Standardization Initiative (ETSI)](https://etsinitiative.org/)
is working towards establishing a standard for PET Raw Data, called PETSIRD ("PET ETSI Raw Data").
More information is on https://github.com/ETSInitiative/PETSIRD.

## Current capabilities
Some additionnal features are following.

### Field of view viewer
An option was added to draw a cylinder, giving the capabilities to the user to also show a field of view.



### How to use this repo

1. Open ***your*** repo in [GitHub Codespaces](https://code.visualstudio.com/docs/remote/codespaces) or
in a [VS Code devcontainer](https://code.visualstudio.com/docs/devcontainers/containers).
This codespace/container will contain all necessary tools, including `yardl` itself, as well as your repository.
2. Use `yardl` to generate C++ and Python code for the model:
  ```sh
  cd YourRepoName
  cd PETSIRD/model
  yardl generate
  cd ../..
  ```
3. Start working in either the [`cpp`](cpp/README.md) and/or [`python`](python/README.md) directories.
