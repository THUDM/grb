# GRB Leaderboard

To better evaluate the *adversarial robustness* of Graph Neural Networks (GNNs), GRB provides *up-to-date* and *reproducible* leaderboards for all involved datasets: [*grb-cora*](https://cogdl.ai/grb/leaderboard/grb-cora), [*grb-citeseer*](https://cogdl.ai/grb/leaderboard/grb-citeseer), [*grb-flickr*](https://cogdl.ai/grb/leaderboard/grb-flickr), [*grb-reddit*](https://cogdl.ai/grb/leaderboard/grb-reddit), [*grb-aminer*](https://cogdl.ai/grb/leaderboard/grb-aminer). To help you understand and reproduce the leaderboard, here are some introductions and instructions.

## How GRB leaderboard is designed?

Unlike other popular leaderboards (e.g. [OGB](https://ogb.stanford.edu/), [Benchmarking GNN](https://github.com/graphdeeplearning/benchmarking-gnns)) that focus on the performance of GNNs, GRB leaderboard aims to tackle the problem of *adversarial robustness*, i.e., the robustness under potential adversarial attacks. Thus, GRB leaderboards consider both adversarial attacks and defenses (GNNs with or without defense mechenisms) and put them under a unified evaluation scenario (introduced in [GRB evaluation rules](https://cogdl.ai/grb/intro/rules)) for all datasets (Statistics and descriptions can be found in [datasets](https://cogdl.ai/grb/datasets)). 

<center>
	<img style="border-radius: 0.3125em;    
              box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     
       src=https://github.com/THUDM/grb/blob/master/docs/source/_static/grb_leaderboard_example.png>    
  <br>    
  <div style="color:black; 1px solid #d9d9d9;    
              display: inline-block;
              padding: 2px;">Example of GRB leaderboard on grb-aminer dataset. (3 attacks vs. 7 vanilla GNNs)</div> 
</center>

In GRB leaderboard, there are attacks shown in vertical line and defenses shown in horizontal line. The value in each cell represents the average accuracy and the standard deviation. All attacks are performed **10 times** with different random seeds to report the error bar. **Easy**/**Medium**/**Hard** refer to different difficulties of the dataset, mainly depending on the average degree of test nodes. **Full** refers to the entire test set. Apart from the accuracy of attack vs. Defense. There are also overall metrics for both attacks and defenses:

Metrics for attacks:

* <span style="color: #a8071a">Avg. Accuracy</span>: Average accuracy of all defenses (including vanilla GNNs and GNNs with defense mechanisms). 
* <span style="color: #a8071a">Avg. 3-Max Accuracy</span>: Average accuracy of the three most robust models (with the highest accuracy).
* <span style="color: #a8071a">Weighted Accuracy</span>: Weighted accuracy of various attacked models, calculated by $s_{w}^{att} = \sum_{i=1}^{n} w_i s_i, w_i = \frac{1/i^2}{\sum_{i=1}^n(1/i^2)}, s_i=(S_{descend}^{def})_i$ where $S_{descend}^{def}$ is the set of defense scores in a descending order. The metric attaches more weight to the most robust defenses.

Metrics for defenses:

*  <span style="color: #10239e">Avg. Accuracy</span>: Average accuracy of all attacks.
* <span style="color: #10239e">Avg. 3-Min Accuracy</span>: Average accuracy of the three most effective attacks (with the lowest accuracy).
* <span style="color: #10239e">Weighted Accuracy</span>: Weighted accuracy of various attacks, calculated by:$s_{w}^{def} = \sum_{i=1}^{n} w_i s_i, w_i = \frac{1/i^2}{\sum_{i=1}^n(1/i^2)}, s_i=(S_{ascend}^{att})_i$ where $S_{ascend}^{att}$ is the set of attack scores in an ascending order. The metric attaches more weight to the most effective attacks.



