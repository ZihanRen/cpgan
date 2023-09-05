## GAN evaluation.
All name with ex6 meaning we are evaluating ex6 model GAN (WGAN-GP). There is no conditional evaluation of GAN. Most scripts focus on checking statistical recosntruction quanlity ensembly. This directory focus on follow purposes:
* Petrophysical property analysis and reconstruction statistics analysis;
* Multiphase flow properties and petrophysical properties sensitive analysis;

Subdirecotory explanation:
* save_obj: saved pickle objects for checking results;
* models_lib: containing all trained models and their history. In ex1 there contains some results analysis
* post_sim_data: PNM simulation results of GAN generated porous media vs real data source


### Petrophysical property analysis and statistics reconstruction
* ex6-eval.ipynb: examine the statistical reconstruction results.