## Citation

Kuodytė, V.; Petkevičius, L. Education to Skill Mapping using Hierarchical Classification and Transformer Neural Network. Applied Sciences 2021.


The publication is available at [Applied Sciences](https://www.mdpi.com/2076-3417/11/13/5868)

## Description

The final model described in the article is included.
Docker container contains parameters of learning environment

### Build docker container:
```sh
git clone https://github.com/Stratosfera/education2skill.git
cd education2skill/
docker build -t stratosfera/education2skill .
```
### Run example

Provided script [model_training.py](./model_training.py) includes script of model training and [prediction.py](./prediction.py) can be used to test how model predicts occupations based on English description of education program.

```sh
docker run -ti education2skill bash
python prediction.py
```

### Acknowledgments
Final note to express sincere gratitude and acknowledge contributions by [@aleksas](https://github.com/aleksas) for indispensable consultations on how to restructure training scripts and docker files for publishing online.

