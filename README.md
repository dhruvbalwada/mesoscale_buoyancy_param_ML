# Data-driven parameterization for mesoscale buoyancy fluxes
## Dhruv Balwada, Pavel Perezhogin, Alistair Adcroft and Laure Zanna
### Lamont-Doherty Earth Observatory, Columbia University/ Courant Institute of Mathematical Sciences, New York University/ Princeton University

Repo shows how results from this paper can be reproduced.


#### Notes for LEAP hub
When running on LEAP hub (Pangeo cloud instances), two libraries are mission (as of 12 Mar 2024):
- `pip install 'flax==0.7.2' 'jax<=0.4.13' 'ml_dtypes==0.2.0' 'jaxlib<=0.4.13'`
- `mamba install cuda-nvcc==11.6.* -c nvidia`
