import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(
    config_path="../config",
    config_name="pretrain.yaml",
    version_base="1.2",
)
def main(config: DictConfig):
    # needed to import fragformer
    import sys
    # sys.path.append('../')
    sys.path.append('/home/d.gaillard/projects/fragment_autoencoder/fragment_foundation_model')

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from fragformer.entrypoints import fragformer_entrypoint
    from fragformer.utils.io_utils import extras, print_config, validate_config

    # Validate config -- Fails if there are mandatory missing values
    validate_config(config)

    # Applies optional utilities
    extras(config)

    if config.get("print_config"):
        print_config(config, resolve=True)

    # Train model
    return fragformer_entrypoint(config)


if __name__ == "__main__":
    main()
