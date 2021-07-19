# Modularity in Reinforcement Learning
This is the code complementing the paper [Modularity in Reinforcement Learning via Algorithmic Independence in Credit Assignment].

## Setup
Set the PYTHONPATH: `export PYTHONPATH=$PWD`.

Create a conda environment with python version 3.6.

Install dependencies: `pip install -r requirements.txt`. 

For GPU, set OMP_NUM_THREADS to 1: `export OMP_NUM_THREADS=1`.

## Training
Run `python runner.py --<experiment-name>` to print out example commands for the environments in the paper. Add the `--for-real` flag to run those commands.

For all experiments, you would need to run the pretraining experiment first (commands are below). Then you would need to specify the expriment folders for all transfer environments using the pre-trained model. You would do this by adding the pretraining folder as an argument to the transfer experiment command for runner.py using the following flag `--parent [pretraining experiment folder]`.

## Experiments:

### Linear Chain: 

#### Pretraining: 
	
PPO:            `python runner.py --linearpretrain_mon --for-real`  
CVS:            `python runner.py --linearpretrain_dec --for-real`  
PPO factorized: `python runner.py --linearpretrain_mon --factorized --for-real`  

#### Transfer: 

PPO: 		`python runner.py --linearrest_mon --parent [pretraining PPO exp folder] --for-real`  
CVS: 		`python runner.py --linearrest_decent --parent [pretraining CVS exp folder] --for-real`  
PPO factorized: `python runner.py --linearrest_mon --factorized --parent [pretraining PPO factorized exp folder] --for-real`  

### Common Ancestor: 

#### Pretraining: 

PPO: `python runner.py --commonancpretrain_mon --for-real`  
CVS: `python runner.py --commonancpretrain_dec --for-real`  
PPO factorized: `python runner.py --commonancpretrain_mon --factorized --for-real`  

#### Transfer: 

PPO: 		`python runner.py --commonancrest_mon --parent [pretraining PPO exp folder] --for-real`  
CVS: 		`python runner.py --commonancrest_decent --parent [pretraining CVS exp folder] --for-real`  
PPO factorized: `python runner.py --commonancrest_mon --factorized --parent [pretraining PPO factorized exp folder] --for-real`  

### Common Descendant: 

#### Pretraining: 
	
PPO:            `python runner.py --commondecpretrain_mon --for-real`  
CVS:            `python runner.py --commondecpretrain_dec --for-real`  
PPO factorized: `python runner.py --commondecpretrain_mon --factorized --for-real`  

#### Transfer: 

PPO: 		`python runner.py --commondecrest_mon --parent [pretraining PPO exp folder] --for-real`  
CVS: 		`python runner.py --commondecrest_decent --parent [pretraining CVS exp folder] --for-real`  
PPO factorized: `python runner.py --commondecrest_mon --factorized --parent [pretraining PPO factorized exp folder] --for-real`

### Bandit: 

#### Pretraining: 
	
PPO:            `python runner.py --bandit --for-real`  
CVS:            `python runner.py --bandit_dec --for-real`  
PPO factorized: `python runner.py --bandit --factorized --for-real`  

#### Transfer: 

PPO: 		`python runner.py --bandittransfer --parent [pretraining PPO exp folder] --for-real`  
CVS: 		`python runner.py --bandittransfer_dec --parent [pretraining CVS exp folder] --for-real`  
PPO factorized: `python runner.py --bandittransfer --factorized --parent [pretraining PPO factorized exp folder] --for-real`

### Invariance (Forgeting) 

#### Pretraining: 
	
PPO:            `python runner.py --invarV1_mon --for-real`  
CVS:            `python runner.py --invarV1_decent --for-real`  
PPO factorized: `python runner.py --invarV1_mon --factorized --for-real`  

#### Transfer V1: 

PPO: 		`python runner.py --invarV2_mon --parent [pretraining PPO exp folder] --for-real`  
CVS: 		`python runner.py --invarV2_decent --parent [pretraining CVS exp folder] --for-real`  
PPO factorized: `python runner.py --invarV2_mon --factorized --parent [pretraining PPO factorized exp folder] --for-real`

#### Transfer V2: 

PPO: 		`python runner.py --invarV1Prime_mon --parent [pretraining PPO exp folder] --for-real`  
CVS: 		`python runner.py --invarV1Prime_decent --parent [pretraining CVS exp folder] --for-real`  
PPO factorized: `python runner.py --invarV1Prime_mon --factorized --parent [pretraining PPO factorized exp folder] --for-real`
