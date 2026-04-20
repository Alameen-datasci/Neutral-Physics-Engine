""" result_analysis.py """
from neutral_physics_engine.analysis import Analysis

# ================================ Result Analysis ================================

filename = "random_n_bodies_realistic.h5"       # Specify the HDF5 file to analyze (e.g., from your simulations)
# You can change this to "simulation_testing_4.h5" or "simulation_testing_6.h5" to analyze those simulations instead.
with Analysis(f"results/{filename}") as f:
    print("Metadata")                   # Print the metadata stored in the HDF5 file (e.g., simulation parameters, integrator used, etc.)
    print(f.get_metadata())             # Get and print the metadata from the HDF5 file for reference.
    print()
    print("Energy Drift Rate")
    print("----" * 10)
    print(f.energy_drift_rate())    # Calculate the energy drift rate, which is a measure of how much the total energy of the system changes over time. A smaller drift rate indicates better energy conservation.
    
    # Plotting various components of the simulation results for analysis and visualization.
    f.relative_energy_error()
    f.plot_energy_components()
    f.plot_linear_momentum()
    f.plot_angular_momentum()
    f.plot_projection(["xy", "yz", "xz"])

# The analysis will provide insights into the energy conservation, momentum conservation, and the trajectories of the bodies in the simulation. You can use these plots and metrics to evaluate the performance of the integrator and the dynamics of the system.
# Note: The specific results and plots will depend on the simulation data stored in the HDF5 file you choose to analyze. You can compare different simulations by analyzing their respective HDF5 files and observing how the energy drift rate and other metrics differ based on the integrator used, the time step, and the initial conditions of the bodies.

# ================================ End of Result Analysis ================================
