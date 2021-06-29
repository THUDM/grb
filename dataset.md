## Scalability

GRB includes five datasets of different scales, *grb-cora*, *grb-citeseer*, *grb-flickr*, *grb-reddit*, *grb-aminer*. The original datasets are gathered from previous works, and in GRB they are reprocessed. The fundamental statistics of these datasets are shown in the statistic table. Besides small-scale datasets which are common in previous works, GRB also includes medium and large-scale datasets for hundreds of thousands of nodes and millions of edges. More details about how the datasets are generated can be found in the following part.

## Splitting Scheme

<center>
   <img style="border-radius: 0.3125em;"
        width="1000"
        src=https://github.com/THUDM/grb/raw/master/docs/source/_static/data_splitting.png>    
  <br>    
  <div style="color:black; 1px solid #d9d9d9;    
              display: inline-block;
              padding: 2px;">Novel splitting scheme for GRB datasets. </div> 
</center>


Random splits are not suitable for a fair comparison across methods, especially when it indeed affects the evaluation results of GNNs ([Shchur et al, 2019](https://arxiv.org/abs/1811.05868)). GRB introduces a new splitting scheme specially designed for evaluating adversarial robustness. The key idea is based on the assumption that nodes with lower degrees are easier to attack, as demonstrated in [Zou et al, 2021](https://arxiv.org/abs/2106.06663). In principle, GNNs aggregate information from neighbor nodes to update a target node. If the target node has few neighbors, it is more likely to be influenced by adversarial perturbations, vice-versa. Thus, we construct test subsets with different average degrees. Firstly, we rank all nodes by their degrees. Secondly, we filter out 5% nodes with the lowest degrees (including isolated nodes that are too easy to attack) and 5% nodes with the highest degree (including nodes connected to hundreds of other nodes that are hardly influenced). Thirdly, we divide the rest nodes into three equal partitions without overlap, and randomly sample 10% nodes (without repetition) from each partition. Finally, we get three test subsets with different degree distributions. According to the average degrees, we define them as Easy/Medium/Hard/Full ('E/M/H/F', 'F' contains all test nodes). For the rest nodes, we divide them into train set (60%) and val set (10%), for training and validation respectively. 

## Feature Normalization

Initially, the features in each dataset have various ranges. To make them in the same scale (e.g. range $[-1, 1]$), we apply a *standardization* following by an *arctan* transformation: $\mathcal{F} = \frac{2}{\pi} arctan(\frac{\mathcal{F} - mean(\mathcal{F})}{std(\mathcal{F})})$. This transformation is bijective, which also permits attackers to restore features in the original feature space to apply real-world adversarial attacks.
