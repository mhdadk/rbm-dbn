# Movie Recommendation using Restricted Boltzmann Machines and Deep Belief Networks

This repository demonstrates the use of Restricted Boltzmann Machines (RBMs) and Deep Belief Networks (DBNs) for movie recommendation. This implementation is based on [this](https://dl.acm.org/doi/10.1145/1273496.1273596) paper. More specifically, Categorical-Bernoulli (Softmax-Bernoulli) and Bernoulli-Bernoulli RBMs are implemented. These implementations can be found in the `models/RBM` directory. The Categorical-Bernoulli RBM is first individually trained, validated, then tested. It is then combined with a Bernoulli-Bernoulli RBM to form a Deep Belief Network.

**IMPORTANT**: It is highly recommended that the user first checks the `RBM.ipynb` Jupyter notebook, as it explains the theory behind RBMs and the code used to implement them.

Repository structure:

* `data`: Contains the data used in `.csv` format. `*2.csv` files are extended versions of the `*.csv` files. All data in the `.csv` files are in the following format: `<A>,<B>,<C>`, where:
  * A is the user number, ranging from 1 to 50 for `*.csv` files and from 1 to 10000 for `*2.csv` files.
  * B is the movie number, ranging from 1 to 50.
  * C is the movie rating, ranging from 1 to 5.
* `diagrams`: Diagrams used for illustration drawn using [diagrams.net](https://www.diagrams.net/).
* `models`: Contains class definitions for the RBMs and DBN.
* `parameters`: Where PyTorch `.pt` files will be saved.
* `torch_datasets`: Contains definitions for the PyTorch `torch.utils.data.dataset` classes. These are used for dataloading.
* `unit_tests`: Contains step-by-step implementations of the RBMs in a format that is more easily debuggable.
* `utils`: Miscellaneous utility functions. 
