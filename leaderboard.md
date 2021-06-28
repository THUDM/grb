# GRB Leaderboard

To better evaluate the *adversarial robustness* of Graph Neural Networks (GNNs), GRB provides *up-to-date* and *reproducible* leaderboards for all involved datasets: [*grb-cora*](https://cogdl.ai/grb/leaderboard/grb-cora), [*grb-citeseer*](https://cogdl.ai/grb/leaderboard/grb-citeseer), [*grb-flickr*](https://cogdl.ai/grb/leaderboard/grb-flickr), [*grb-reddit*](https://cogdl.ai/grb/leaderboard/grb-reddit), [*grb-aminer*](https://cogdl.ai/grb/leaderboard/grb-aminer). All GNNs, adversarial attacks, and defenses can be fairly compared under a unified evaluation scenario, which facilitates researchers to evaluate the effectiveness of newly proposed methods and promote future research in this area. To help you understand and reproduce the leaderboard, here are some introductions and instructions.

## How GRB leaderboard is designed?

Unlike other popular leaderboards (e.g. [OGB](https://ogb.stanford.edu/), [Benchmarking GNN](https://github.com/graphdeeplearning/benchmarking-gnns)) that focus on the performance of GNNs, GRB leaderboard aims to tackle the problem of *adversarial robustness*, i.e., the robustness under potential adversarial attacks. Thus, GRB leaderboards consider both adversarial attacks and defenses (GNNs with or without defense mechanisms) and put them under a unified evaluation scenario (introduced in [GRB evaluation rules](https://cogdl.ai/grb/intro/rules)) for all datasets (Statistics and descriptions can be found in [datasets](https://cogdl.ai/grb/datasets)). 

<center>
	<img style="border-radius: 0.3125em;    
              box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     
       src=https://github.com/THUDM/grb/raw/master/docs/source/_static/grb_leaderboard_example.png>    
  <br>    
  <div style="color:black; 1px solid #d9d9d9;    
              display: inline-block;
              padding: 2px;">Example of GRB leaderboard on grb-aminer dataset. (3 attacks vs. 7 vanilla GNNs)</div> 
</center>


In GRB leaderboard, the attacks are shown vertically and the defenses horizontally. The value in each cell represents the average accuracy and the standard deviation. All attacks are performed **10 times** with different random seeds to report the error bar. **Easy**/**Medium**/**Hard** refer to different difficulties of the dataset, mainly depending on the average degree of test nodes, **Full** refers to the entire test set, more explanation can be found in [datasets](https://cogdl.ai/grb/datasets). Apart from each attack vs. defense accuracy, there are also overall metrics for both attacks and defenses:

**Metrics for Attacks**:

* <span style="color: #a8071a">Avg. Accuracy</span>: Average accuracy of all defenses (including vanilla GNNs and GNNs with defense mechanisms). 
* <span style="color: #a8071a">Avg. 3-Max Accuracy</span>: Average accuracy of the three most robust models (with the highest accuracy).
* <span style="color: #a8071a">Weighted Accuracy</span>: Weighted accuracy of various attacked models, calculated by $s_{w}^{att} = \sum_{i=1}^{n} w_i s_i, w_i = \frac{1/i^2}{\sum_{i=1}^n(1/i^2)}, s_i=(S_{descend}^{def})_i$ where $S_{descend}^{def}$ is the set of defense scores in descending order. The metric attaches more weight to the most robust defenses.

**Metrics for Defenses**:

*  <span style="color: #10239e">Avg. Accuracy</span>: Average accuracy of all attacks.
* <span style="color: #10239e">Avg. 3-Min Accuracy</span>: Average accuracy of the three most effective attacks (with the lowest accuracy).
* <span style="color: #10239e">Weighted Accuracy</span>: Weighted accuracy of various attacks, calculated by:$s_{w}^{def} = \sum_{i=1}^{n} w_i s_i, w_i = \frac{1/i^2}{\sum_{i=1}^n(1/i^2)}, s_i=(S_{ascend}^{att})_i$ where $S_{ascend}^{att}$ is the set of attack scores in ascending order. The metric attaches more weight to the most effective attacks.

**Comparison Mode**:

<center>
	<img style="border-radius: 0.3125em;    
              box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     
       src=https://github.com/THUDM/grb/raw/master/docs/source/_static/grb_leaderboard_comparison.png>    
  <br>    
  <div style="color:black; 1px solid #d9d9d9;    
              display: inline-block;
              padding: 2px;">Example of comparison mode (with GCN chosen) on grb-aminer dataset. </div> 
</center>

* Click on the *compare* button right close to each method to enter the comparison mode.
* In this mode, the chosen method (either attack or defense) can be easily compared with all other methods. The **bold** score indicates that the chosen method is significantly better/worse than the compared baseline, under a t-test setting with a *p-value* at 0.05. 

**Customizable Configurations**:

<center>
	<img style="border-radius: 0.3125em;    
              box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     
       src=https://github.com/THUDM/grb/raw/master/docs/source/_static/grb_leaderboard_configuration.png>    
  <br>    
  <div style="color:black; 1px solid #d9d9d9;    
              display: inline-block;
              padding: 2px;">Customizable configurantion of GRB leaderboard. </div> 
</center>

* Users are allowed to customize the configuration to compare various **Attacks** or **Models** under different **Difficulty**.
* There are also four preset configurations:
  * **Brief**: The best 3 attacks and the best 5 defenses under the **Full** difficulty.
  * **Main**: The best 5 attacks and the best 10 defenses under the **Full** difficulty. Default configuration.
  * **Completed**: All attacks and all defenses under all four difficulties. 
  * **No Defense**: All attacks and all vanilla GNNs (without any defense mechanism).

**Ranking Bar Chart**:

* Bar charts are provided to visualize the ranking. 
* The ranking is determined by the weighted accuracy for both attacks and defenses (under **Full** difficulty).

With the above designs, GRB leaderboard is able to track the progress on impoving the *adversarial robustness* of GNNs, by identifying the most robust models and the most effective adversarial attacks/defenses. 

## How to reproduce results on GRB leaderboard?

GRB respects high reproducibility for all results on the leaderboards. To this end, GRB provides all necessary information to reproduce the results including datasets, implementations, attack results, saved models, etc. GRB also has a modular coding framework containing Implementations of all methods, which can be found in the Github repo: [https://github.com/THUDM/grb](https://github.com/THUDM/grb). There are also scripts to reproduce the leaderboards. Here are some instructions:

1.  Install ``grb`` package:

   ```bash
   git clone git@github.com:THUDM/grb.git
   cd grb
   pip install -e .
   ```

2. Get datasets from [link](https://cloud.tsinghua.edu.cn/d/c77db90e05e74a5c9b8b/) or download them by running the following script:
   ```bash
   cd ./scripts
   sh download_dataset.sh
   ```
   Get attack results (adversarial adjacency matrix and features) from [link](https://cloud.tsinghua.edu.cn/d/94b2ea104c2e457d9667/) or download them by running the following script:
   ```bash
   sh download_attack_results.sh
   ```
   Get saved models (model weights) from [link](https://cloud.tsinghua.edu.cn/d/8b51a6b428464340b368/) or download them by running the following script:
   ```bash
   sh download_saved_models.sh
   ```

3. Run the following script to reproduce results on leaderboards:

   ```bash
   sh run_leaderboard_pipeline.sh -d grb-cora -g 0 -s ./leaderboard -n 0
   Usage: run_leaderboard_pipeline.sh [-d <string>] [-g <int>] [-s <string>] [-n <int>]
   Pipeline for reproducing leaderboard on the chosen dataset.
       -h      Display help message.
       -d      Choose a dataset.
       -s      Set a directory to save leaderboard files.
       -n      Choose the number of an attack from 0 to 9.
       -g      Choose a GPU device. -1 for CPU.
   ```

Users are allowed to add new methods or current methods with different hyperparameters by modifying configurations in ``./pipeline/grb-xxx/config.py``. For future submissions, users should follow the [GRB Evaluation Rules](https://cogdl.ai/grb/intro/rules) to ensure the reproducibility. The submission platform will be ready soon. 
