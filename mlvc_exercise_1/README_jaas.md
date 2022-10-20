### Install Conda Environment

1. conda create --name MLVC
2. conda init
3. source ~/.bashrc
4. conda activate MLVC
5. conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
6. conda install -c conda-forge opencv matplotlib tqdm torchinfo pandas
7. conda install nbformat ipykernel
8. python -m ipykernel install --user --name=MLVC

### Frequently Asked Questions

### Known Bugs

* Currently, there is a bug with the jupyter notebook from TUWEL, that does not activate conda in the terminal.
    ---> Current Workaround: Use source ~/.bashrc to get access to conda
* Another issue is that conda environments are disappearing and installed jupyter notebook kernels stop working (imports fail)
    ---> Current Workaround: Reinstall the environment (steps 1 to 8) or reinstalling the kernel (step 8)
The JAAS TUWEL Team knows about these issues and is looking for fixes.
