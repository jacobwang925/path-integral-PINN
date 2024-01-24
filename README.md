# path-integral-PINN

An implementation of paper "Physics-informed Representation and Learning: Control and Risk Quantification", accepted at the AAAI 24 conference.



## How to Run

The simulation for path integral MC is implemented in MATLAB. The version used for development is R2023a.

The simulation for physcis-informed neural network (PINN) and autoencoder-like NN is implemented in Python. The packages used for development is listed in `requirement.txt`

To run the code, consider creating a virtual environment using [conda](https://www.anaconda.com/) via the following command:

```bash
conda create --name pinn
conda activate pinn
```

The simulation uses [DeepXDE](https://arxiv.org/abs/1907.04502) to construct the physics-informed neural networks. See [official document](https://deepxde.readthedocs.io/en/latest/) for installation details.



### Path Integral Monte Carlo (MC)

For results on value function estimation, run `value_function_1kd.m`

For sample complexity analysis, run  `sample_complexity_3d.m`.

For visualization of the sample complexity analysis, run  `value_function_3d.m`.

For results on safety probability estimation, run  `safety_probability_3d.m`.



### Physics-informed Neural Network (PINN)

For value function estimation results on 1000d system, run

```bash
python src/PINN/value_function_pde_dim1000.py
```

or execute [value_function_PDE_1000d.ipynb](https://colab.research.google.com/drive/150h7qsD0H5k4MHqbEe-RwOkmwbq02oEe#scrollTo=Oqs1r8YqY7uE) on Google Colab.



For value function estimation results on 3d system, run

```bash
python src/PINN/value_function_pde_dim3.py
```

or execute [value_function_PDE_3d.ipynb](https://colab.research.google.com/drive/1UI_UhJBcGr-Y_v4ES0MyQ-DM3YGbNdrK#scrollTo=eZW5wVaMYSHC) on Google Colab.



For safety probability estimation results on 3d system, run

```bash
python src/PINN/safety_probability_dim3.py
```

or execute [safety_prob_PDE_3d.ipynb](https://colab.research.google.com/drive/1_qdmKX6u-eRwRi73xR19oX5RhF_BZW49#scrollTo=HYHgzuP6YOKn) on Google Colab.



### Autoencoder-like Neural Network

Functions to compute the preimage can be found in `utils.py`. 

To train the model, set the desired parameters in `trainer.py` and run

```bash
python src/autoencoder/trainer.py
```



## Citation

```
@article{wang2023physics,
  title={Physics-informed Representation and Learning: Control and Risk Quantification},
  author={Wang, Zhuoyuan and Keller, Reece and Deng, Xiyu and Hoshino, Kenta and Tanaka, Takashi and Nakahira, Yorie},
  journal={arXiv preprint arXiv:2312.10594},
  year={2023}
}
```

