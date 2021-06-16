# GRB Evaluation Rules

## Evaluation Scenario

![GRB](https://github.com/THUDM/grb/blob/master/docs/source/_static/grb_scenario.png)

GRB provides a unified evaluation scenario for fair comparisons between attacks and defenses. The scenario is **Black-box**, **Evasion**, **Inductive**, **Injection**. Take the case of a citation-graph classification system for example. The platform collects labeled data from previous papers and trains a GNN model. When a batch of new papers are submitted, it updates the graph and uses the trained model to predict labels for them. 

* **Black-box**: Both the attacker and the defender have no knowledge about the applied methods each other uses.
* **Evasion**: GNNs are already trained in trusted data (e.g. authenticated users), which are untouched by the attackers but might have natural noises. Thus, attacks will only happen during the inference phase. 
* **Inductive**: GNNs are used to classify unseen data (e.g. new users), i.e. validation or test data are unseen during training, which requires GNNs to generalize to out of distribution data.
* **Injection**: The attackers can only inject new nodes but not modify the target nodes directly. Since it is usually hard to hack into users' accounts and modify their profiles. However, it is easier to create fake accounts and connect them to existing users.

## Rules of Evaluation

We clarify attacker and defender's capability in the following:

* **For attackers**: they have knowledge about the entire graph (including all nodes, edges and labels, **excluding** labels of the test nodes to attack), but do **NOT** have knowledge about the target model or the defense mechanism; they are allowed to inject a limited number of new nodes with limited edges, but are **NOT** allowed to modify the original graph; they are allowed to generate features of injected nodes as long as they remain **unnoticeable** by defenders (e.g. nodes with features that exceed the range can be easily detected); they are allowed to get the classification results from the target model through limited number of queries.  
* **For defenders**: they have knowledge about the entire graph **excluding** the test nodes to be attacked (thus only the training and validation graph); they are allowed to use any method to increase adversarial robustness, but **NOT** having prior knowledge about what kind of attack is used or about which nodes in the test graph are injected nodes.

Besides, it is reasonable that both sides can make assumptions even in **Black-box** scenario. For example, the attackers can assume that the GNN-based system uses GCNs, since it is one of the most popular GNNs. Also, it is not reasonable to assume that the defense mechanism can be completely held secretly, known as the [Kerckhoffs’ principle](https://en.wikipedia.org/wiki/Kerckhoffs%27s_principle). If a defense wants to be general and universal, it should guarantee part of robustness even when attackers have some knowledge about it. 
