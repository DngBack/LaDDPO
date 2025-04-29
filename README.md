# LaDDPO

LaDDPO (Language Diffusion DPO)

### Set up

#### Set up env

Using python 3.11

```bash
conda create -n laddpo python=3.11 -y

conda activate laddpo
```

#### Set up pre-commit to format code

    - Install:
    ```bash
    pip install pre-commit
    ```

    - Add pre-commit to git hook:
    ```bash
    pre-commit install
    ```

    - Run pre-commit for formating code (only staged files in git):
    ```bash
    pre-commit run
    ```

    - Run pre-commit for formating code with all files:
    ```bash
    pre-commit run --all-files
