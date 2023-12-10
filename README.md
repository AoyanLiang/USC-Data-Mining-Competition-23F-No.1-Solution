# USC-Data-Mining-Competition-23F-No.1-Solution
## Method Description:

- Feature Engineering:
(1) Graph embedding
To enhance model accuracy beyond a threshold of RMSE 0.98, I explored various feature engineering techniques. Initially, word embeddings were applied to convert user review texts into vectors, but this approach did not significantly impact performance due to the limited size and scope of the input data (user and business IDs). Subsequently, graph embedding was employed, leveraging the inherent graph-structured nature of some dataset features. Relationships like 'friend' between users, 'reviewed' between users and businesses, and 'in' or 'belong' for businesses with cities and categories were transformed into graph embeddings. Traditional methods like node2vec were unsuitable for our large graph comprising over 65 million edges. Therefore, I utilized PyTorch-BigGraph (PBG), an efficient tool for handling graphs with billions of edges. The embedding was executed on the USC CARC with a 64 CPU node, resulting in 100-dimensional vectors for user IDs, business IDs, cities, and categories. For businesses with multiple categories, their embeddings were averaged.

PBG: https://github.com/facebookresearch/PyTorch-BigGraph

(2) Other features:
Additional features were derived from the user and business JSON files. Certain categorical and string features were dropped, while others were converted into numerical formats. The feature_process_noreview function outlines these transformations.

- Hyperparameter tunning:
Bayesian optimization was employed to fine-tune the XGBoost model's hyperparameters, offering greater efficiency than traditional grid or random search methods. The optimization focused solely on the training dataset, with the validation set used for early stopping to prevent overfitting. I used Optuna for such process. After 13 iterations, the validation dataset achieved an RMSE of 0.971097, at which point further optimization was deemed cost-prohibitive. The tuning process and results are documented in train.ipynb.

- Model selection and training:
Various models were evaluated, with gradient decision trees demonstrating superior performance. Although XGBoost, CatBoost, and LightGBM yielded comparable results, XGBoost was selected for its balance of efficiency and effectiveness. Using the optimal hyperparameters identified through Bayesian optimization, the model was trained on a combined set of training and validation data. This approach aimed to enhance accuracy on both validation and test datasets. The training procedure and model details are available in train.ipynb.

Error Distribution:
>=0 and <1: 108400
>=1 and <2: 28780
>=2 and <3: 4532
>=3 and <4: 332
>=4: 0

RMSE for validation dataset:
0.8909814953804016

RMSE for test dataset:
0.9680267140533286

Execution Time:
- PBG graph embedding: 1.5 hours
- Hyperparameter tuning for XGBoost using Bayesian optimization: 6 hours for 13 trials
- Model training with best hyperparameters: 1.5 hours
- Model prediction for validation dataset (including feature processing): ~270 seconds
