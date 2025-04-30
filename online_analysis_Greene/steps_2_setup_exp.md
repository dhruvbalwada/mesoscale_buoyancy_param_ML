# Setting up a suite of MOM6 experiments

Sometimes for parameter sensitivity studies, we may want to setup a large suite of experiments where some parameter values are systematically varied. Here provide some steps for how this may be done. 

These are very custom steps for dummies, so you don't forget anything.
1. Create a clean experiment template tailored to the runs you want.
    - This should have most of settings that we are not looking to change.
    - You may call this `exp_name_clean`.
    - Files that need to be examined `INPUT` (may contain ANN weights), `mom.sub`, `diag_table`, `input.nml`, `MOM_override`.
    - For the `diag_table` think about how many days you are expecting to run the simulation, which will set the number in file names `prog4dy`. And also about what variables may be needed for the diagnostics of interest.
2. Test that the clean experiment template simulation runs as expected.
    - Check how fast the simulation runs, and if everything looks good.
    - You may call this `exp_name_run`.
    - One the exp runs, check using a notebook or something similar.
3. 
  

