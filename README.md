# UNIMIB SNOWIT PROJECT
This project is prepared using the package manager PDM (https://pdm-project.org/).<br>
It is recommended to use a virtual environment, it is used Conda (https://docs.anaconda.com/free/miniconda/miniconda-install/).<br>
It is recommended to use an IDE for developing your code, it is used Visual Studio Code (https://code.visualstudio.com/).<br>
It is recommended to use git for version control (https://git-scm.com/).<br>

Please update the ```pyproject.toml``` file accordingly.<br>


To setup the project (after installed pdm and conda) follow these step:
1. Create an environment and activate it
    ```
    conda create -n <ENV NAME> python=3.12
    conda activate <ENV NAME>
    ```
2. Go to the project folder where the pyproject.toml is and run
    ```
    cd <PATH PROJECT FOLDER>\unimib_snowit_project
    pdm install
    ```

