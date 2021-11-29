# QLTSP_Thesis
This repository contains the software implemented for the thesis entitled __Applying Quantum Annealing to an extension of the Traveling Salesperson Problem: Implementation and resulting Methodology__. Since the case study problem consists of a real-world application that comes from the industry, we had to remove the classes directly acting on the real dataset which remains proprietary. In this repository we have both the Notebooks used to test the implementation of the algorithm that solves the LTSP (in folder __Model_Testing__), and the whole quantum software (in folder __src__).

## Structure of this Repository
The structure of this repository is organized as follows:

<pre>

├─── Model_Testing
│   ├─── Test_Problems
│   │   ├─── last_tests
│   │   │   ├─── ltsp0.txt
│   │   │   ├─── ...
│   │   │   └─── ltsp4.txt
│   │   │   
│   │   ├─── ltsp0.txt
│   │   ├─── ...          
│   │   └─── ltsp10.txt                     
│   ├─── Basic_Problem.ipynb 
│   ├─── Model_Weights.ipynb
│   └─── Model_Charges.ipynb
│
└─── src
    ├─── code
    │   ├─── model
    │   │   ├─── experiments
    │   │   │   ├─── single_experiment.py
    │   │   │   └─── experiment.py
    │   │   └─── model.py
    │   │
    │   ├─── preprocessing
    │   │   └─── preprocessing.py
    │   │  
    │   ├─── quantum_annealing
    │   │   ├─── performance
    │   │   │   ├─── results_perfornance.py
    │   │   │   └─── time_performance.py
    │   │   │
    │   │   ├─── results
    │   │   │   ├─── result.py
    │   │   │   ├─── results_analyzer.py
    │   │   │   ├─── results_visualizer.py
    │   │   │   └─── results_writer.py
    │   │   │
    │   │   ├─── tuning 
    │   │   │   ├─── abc_tuning.py
    │   │   │   ├─── chain_strength_tuning.py
    │   │   │   └─── anneal_schedule_tuning.py
    │   │   │
    │   │   ├─── model_builder.py
    │   │   ├─── SA_model.py
    │   │   ├─── QPU_model.py
    │   │   ├─── Hybrid_model.py
    │   │   └─── hamiltonian_builder.py
    │   │
    │   ├─── main.py
    │   └─── qltsp.py
    │
    ├─── dataset
    │   └─── parse_experiments.py
    │
    └─── resources 
        ├─── client_config.con
        └─── experiments.json

</pre>
