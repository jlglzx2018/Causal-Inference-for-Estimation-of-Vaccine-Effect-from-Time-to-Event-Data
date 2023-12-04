# Causal-Inference-for-Estimation-of-Vaccine-Effect-from-Time-to-Event-Data
Vaccine is the most efficient method for controlling of infectious disease. Vaccine effectiveness estimation is extremely important in monitoring vaccine efficacy and controlling disease spreading. We find that great challenges arise in vaccine effectiveness estimation due to various sources of biases, including confounding/treatment selection bias, censoring bias, healthy vaccinee bias, event-induced covariate shifts, competing events, prior immunity, emergence of new virus variants, healthcare seeking behavior. A vital step in improving vaccine effectiveness studies is to develop novel methods to address bias. To achieve this goal, we adapt recently developed generalization bounds from domain adaptation and treatment effect estimation to time-to-event data, and apply the counterfactual causal inference method with deep neural network-based balancing representations and the framework of empirical risk minimization (ERM), which is called survITE [1], to vaccine effectiveness estimation from the time-to-event data which are extracted from Optum® de-identified COVID-19 Electronic Health Record dataset (2007-2021). 
The parameters in the code were modified according to specific dataset. 
The estimated vaccine effectiveness by the survITE is compared with the Cox regression model and Random survival forest model. The detailed methods is available in manuscript: Causal Inference for Estimation of Vaccine Effect from Time-to-Event Data.

survITE folder contains the main code of survITE model. 
RSF.py is the code for random survival forest model.
cox_ph.R is the model for cox regression [2].

Reference:
[1] Curth, Alicia, Changhee Lee, and Mihaela van der Schaar. "Survite: Learning heterogeneous treatment effects from time-to-event data." Advances in Neural Information Processing Systems 34 (2021): 26740-26753.
[2] Lin, Dan-Yu, et al. "Effectiveness of Covid-19 vaccines over a 9-month period in North Carolina." New England Journal of Medicine 386.10 (2022): 933-941.

