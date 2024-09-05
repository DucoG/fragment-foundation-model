# Fragment Foundation Model

This repository contains the code and configuration files for training a transformer-based foundation model for fragmentomics.

## Project Structure

- `config/`: Contains YAML configuration files for various components of the project, including data modules, models, trainers, and machine settings.
- `fragformer/`: The core package containing the implementation of the foundation model, including:
  - `datamodule/`: Data loading and processing modules
  - `lit_modules/`: PyTorch Lightning modules
  - `models/`: Model architectures
  - `transforms/`: Data transformation and augmentation
  - `utils/`: Utility functions
- `tools/`: Contains scripts for training, testing, and other utilities.

## Training

To start training the model:

```python tools/pretrain.py```

For distributed training on a SLURM cluster, use:

```sbatch tools/pretrain_fragformer.sh```

## Configuration

The project uses Hydra for configuration management. You can modify the configurations in the `config/` directory to adjust model parameters, data processing, and training settings.

## License

This project is licensed under the terms included in the `LICENCE.txt` file.

## Contact

For questions or feedback, please open an issue in this repository.
