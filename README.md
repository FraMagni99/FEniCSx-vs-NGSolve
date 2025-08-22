# FEniCSx-vs-NGSolve
Test problems run both with FEniCSx and NGSolve. Used to better choose which library will be more convenient in the future.

The files for NGSolve have been uploaded as .ipynb files in order to have the possibility of looking at the plots of the solutions both with webgui, i.e., directly on the notebook, and with external softwares, e.g., ParaView.

Observations:
- on FEniCSx the boundaries can only be exported as piecewise linear. If we want better accuracy, we need to import the mesh from an external software like Gmsh;
- on NGsolve it's easy to represent the solution with curved boundaries using webgui. However, when the solution is saved in .vtu format (to have .xdmf files a further installation is required) the edges are exported as piecewise linear. It's possible to deal with this by refining the elements in the visualisation procedure (and not in the computations), but edges are still kept straight.
- on NGSolve, if we want to impose boundary conditions on some components of the solution, the GridFunction must be saved as a list of scalar quantities. Thus, it's necessary to compose them (e.g., $u_x * \hat{\boldsymbol{i}} + u_y * \hat{\boldsymbol{j}}$) on ParaView to how the trend of the whole solution.
