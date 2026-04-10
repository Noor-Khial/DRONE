[README.md](https://github.com/user-attachments/files/26626356/README.md)
# DRONE-RL

Code for the paper **DRONE-RL: Dynamic Reinforcement Learning for Online Navigation of UAVs in Evolving Environments**.

A hybrid UAV navigation framework that (1) pre-trains a set of DDPG policies offline, each on a different environment configuration, then (2) uses an EXP3-based online learner at deployment to adaptively select among those policies in real time, tracking the optimal in-hindsight policy under arbitrary environmental changes — with provably sublinear regret.

## Citation

```bibtex
@article{KHIAL2026115147,
  title   = {DRONE-RL: Dynamic reinforcement learning for online navigation of UAVs in evolving environments},
  journal = {Knowledge-Based Systems},
  volume  = {334},
  pages   = {115147},
  year    = {2026},
  doi     = {https://doi.org/10.1016/j.knosys.2025.115147},
  url     = {https://www.sciencedirect.com/science/article/pii/S095070512502180X},
  author  = {Noor Khial and Mhd Saria Allahham and Naram Mhaisen and Loay Ismail and Mohamed Mabrok and Amr Mohamed}
}
```
