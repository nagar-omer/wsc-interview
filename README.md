# wsc-interview

This Repo contains implementation for

1. EDA for Action-Phrase dataset
2. PyTorch implementation for BERT based model for Action-Phrase binary classification
3. Lightning based training script
4. Inference system for the trained model

file structure:

```
|-- wsc-interview
    |-- README.md
    |-- scripts
    |   |-- config_yamls
    |   |   |-- config_1.yaml
    |   |   |-- config_2.yaml
    |   |   |-- config_3.yaml
    |   |-- action_inference.py                     - script for inference
    |   |-- perform_eda.py                          - script for EDA
    |   |-- train_action_model.py                   - script for training
    |-- wsc_interview
        |-- models
        |   |-- action_classification_model.py      - model definition and lightning weapper
        |   |-- bert.py                             - huggningface BERT model tools
        |   |-- data_loader.py                      - data loader for training / inference
        |-- utils
        |   |-- eda_utils.py                        - EDA utilities
        |   |-- utils.py                            - general utilities
    |-- setup.py                                    - setup file for package
    |-- requirements.txt                            - requirements file
    
    |-- test                                        - PyTest test cases (only partial coverage)
        |-- ...
```