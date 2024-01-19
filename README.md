# path-integral-PINN

An implementation of paper "Physics-informed Representation and Learning: Control and Risk Quantification", accepted at the AAAI 24 conference.



## How to Run

The simulation for path integral MC is implemented in MATLAB. The version used for development is R2023a.

The simulation for PINN and autoencoder-like NN is implemented in Python. The packages used for development is listed in `requirement.txt`

To run the code, consider creating a virtual environment using [Conda](https://www.anaconda.com/) via the following command:

```bash
conda create --name pinn --file requirements.txt
conda activate pinn
```



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

For value function estimation results on 3d system, run

```bash
python src/PINN/value_function_pde_dim3.py
```

For safety probability estimation results on 3d system, run

```bash
python src/PINN/safety_probability_dim3.py
```



### Autoencoder-like Neural Network

Functions to compute the preimage can be found in `utils.py`. 

To train the model, set the desired parameters in `trainer.py` and run

```bash
python src/autoencoder/trainer.py
```

## 

## Citation

```
@article{wang2023physics,
  title={Physics-informed Representation and Learning: Control and Risk Quantification},
  author={Wang, Zhuoyuan and Keller, Reece and Deng, Xiyu and Hoshino, Kenta and Tanaka, Takashi and Nakahira, Yorie},
  journal={arXiv preprint arXiv:2312.10594},
  year={2023}
}
```

