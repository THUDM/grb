* All experiments are repeated 10 times with different attack seeds.

* Metric for attacks: <span style="color: #a8071a">Avg. Accuracy</span>, <span style="color: #a8071a">Avg. 3-Max Accuracy</span>, <span style="color: #a8071a">Weighted Accuracy</span>.

* Metric for defenses: <span style="color: #10239e">Avg. Accuracy</span>, <span style="color: #10239e">Avg. 3-Min Accuracy</span>, <span style="color: #10239e">Weighted Accuracy</span>.

* Avg. 3-<span style="color: #a8071a">Min</span>/<span style="color: #10239e">Max</span> Accuracy: average accuracy of three most effective attacks or three most robust models.

* Weighted accuracy: attach higher weight for more effective attacks / more robust models.

* In comparison mode, a **bold** score indicates that the target model/attack is significantly better/worse than the compared baseline, under a t-test setting with p-value at 0.05.

* <span style="color: #ff9c6e">AT</span>: adversarial training; <span style="color: #1890ff">LN</span>: layer normalization.