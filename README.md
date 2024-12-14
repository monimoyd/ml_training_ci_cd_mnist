![Build Status](https://github.com/monimoyd/ml_training_ci_cd_mnist/actions/workflows/ml-pipeline.yml/badge.svg)

# Steps to run locally:
## 1. Create a virtual environment:
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
## 2. Install dependencies and run the pytest to unit test model parameters below 20K , use of batch normalization, dropout and GAP:
bash
pip install -r requirements.txt
pytest tests/test_model.py # On Windows: pytest tests\test_model.py

## 3. Train the model and check for test accuracy > 99.4  within 20K parameters and display training logs:

Train the notebook file mnist_train_notebook.ipynb in collab

The best test accuracy achieved is: 99.44%  at 19th epoch

The training logs are pasted below:


Epoch:  1

loss=0.10725033283233643 batch_id=468: 100%|██████████| 469/469 [00:37<00:00, 12.53it/s]


Test set: Average loss: 0.0577, Accuracy: 9804/10000 (98.04%)

Epoch:  2

loss=0.04507721588015556 batch_id=468: 100%|██████████| 469/469 [00:32<00:00, 14.36it/s]


Test set: Average loss: 0.0350, Accuracy: 9893/10000 (98.93%)

Epoch:  3

loss=0.04919236898422241 batch_id=468: 100%|██████████| 469/469 [00:32<00:00, 14.44it/s]


Test set: Average loss: 0.0307, Accuracy: 9897/10000 (98.97%)

Epoch:  4

loss=0.0721362754702568 batch_id=468: 100%|██████████| 469/469 [00:32<00:00, 14.23it/s]


Test set: Average loss: 0.0315, Accuracy: 9898/10000 (98.98%)

Epoch:  5

loss=0.0359935499727726 batch_id=468: 100%|██████████| 469/469 [00:32<00:00, 14.63it/s]


Test set: Average loss: 0.0260, Accuracy: 9917/10000 (99.17%)

Epoch:  6

loss=0.023372627794742584 batch_id=468: 100%|██████████| 469/469 [00:32<00:00, 14.41it/s]


Test set: Average loss: 0.0236, Accuracy: 9928/10000 (99.28%)

Epoch:  7

loss=0.06399094313383102 batch_id=468: 100%|██████████| 469/469 [00:34<00:00, 13.74it/s]


Test set: Average loss: 0.0218, Accuracy: 9938/10000 (99.38%)

Epoch:  8

loss=0.00913456454873085 batch_id=468: 100%|██████████| 469/469 [00:33<00:00, 14.05it/s]


Test set: Average loss: 0.0248, Accuracy: 9924/10000 (99.24%)

Epoch:  9

loss=0.05936382710933685 batch_id=468: 100%|██████████| 469/469 [00:32<00:00, 14.53it/s]


Test set: Average loss: 0.0229, Accuracy: 9926/10000 (99.26%)

Epoch:  10

loss=0.01699383743107319 batch_id=468: 100%|██████████| 469/469 [00:31<00:00, 14.89it/s]


Test set: Average loss: 0.0201, Accuracy: 9928/10000 (99.28%)

Epoch:  11

loss=0.03836320340633392 batch_id=468: 100%|██████████| 469/469 [00:31<00:00, 14.69it/s]


Test set: Average loss: 0.0217, Accuracy: 9934/10000 (99.34%)

Epoch:  12

loss=0.003786402754485607 batch_id=468: 100%|██████████| 469/469 [00:31<00:00, 14.67it/s]


Test set: Average loss: 0.0201, Accuracy: 9936/10000 (99.36%)

Epoch:  13

loss=0.011765461415052414 batch_id=468: 100%|██████████| 469/469 [00:32<00:00, 14.61it/s]


Test set: Average loss: 0.0222, Accuracy: 9925/10000 (99.25%)

Epoch:  14

loss=0.0203243400901556 batch_id=468: 100%|██████████| 469/469 [00:31<00:00, 14.98it/s]


Test set: Average loss: 0.0211, Accuracy: 9933/10000 (99.33%)

Epoch:  15

loss=0.06036381050944328 batch_id=468: 100%|██████████| 469/469 [00:32<00:00, 14.63it/s]


Test set: Average loss: 0.0193, Accuracy: 9933/10000 (99.33%)

Epoch:  16

loss=0.03367232158780098 batch_id=468: 100%|██████████| 469/469 [00:32<00:00, 14.64it/s]


Test set: Average loss: 0.0222, Accuracy: 9933/10000 (99.33%)

Epoch:  17

loss=0.025960639119148254 batch_id=468: 100%|██████████| 469/469 [00:31<00:00, 14.82it/s]


Test set: Average loss: 0.0204, Accuracy: 9936/10000 (99.36%)

Epoch:  18

loss=0.007040448486804962 batch_id=468: 100%|██████████| 469/469 [00:32<00:00, 14.63it/s]


Test set: Average loss: 0.0209, Accuracy: 9938/10000 (99.38%)

Epoch:  19

loss=0.0126817487180233 batch_id=468: 100%|██████████| 469/469 [00:32<00:00, 14.33it/s]


Test set: Average loss: 0.0208, Accuracy: 9944/10000 (99.44%)


## 4. To deploy to GitHub:

* Create a new repository on GitHub

* Initialize local git repository:

bash
git init

git add .

git commit -m "Initial commit"

git branch -M main

git remote add origin https://github.com/monimoyd/ml_training_ci_cd_mnist.git

git push -u origin main

## 5. Running GitHub Actions:

### The GitHub Actions workflow will automatically:

* Set up a Python environment

* Install dependencies

* Run all tests


## 6. The tests check for:

i. Model parameter count (< 20000)

ii. Use of Batch Normalization in Model

iii. Use of DropOut in Model

iv. Use of GAP in Model










