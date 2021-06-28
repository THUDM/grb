* **Metric for Attacks**: <span style="color: #a8071a">Avg. Accuracy</span>, <span style="color: #a8071a">Avg. 3-Max Accuracy</span>, <span style="color: #a8071a">Weighted Accuracy</span>.
* **Metric for Defenses**: <span style="color: #10239e">Avg. Accuracy</span>, <span style="color: #10239e">Avg. 3-Min Accuracy</span>, <span style="color: #10239e">Weighted Accuracy</span>.
* **Avg. 3-<span style="color: #10239e">Min</span>/<span style="color: #a8071a">Max</span> Accuracy**: average accuracy of three most effective attacks or three most robust models.
* **Weighted Accuracy**: attach higher weight for more effective attacks or more robust models.
* **Comparison Mode**: Click on the *compare* button closed to each method to enter this mode, a **bold** score indicates that the target model/attack is significantly better/worse than the compared baseline, under a t-test setting with a *p-value* at 0.05.
* **Randomness**: All experiments are repeated 10 times with different attack random seeds to report the error bar (models are unchanged). 
* **Ranking**: All attacks/defenses are ranked according to **Weighted Accuracy**.
* <span style="color: #ff9c6e">AT</span>: Adversarial Training; <span style="color: #1890ff">LN</span>: Layer Normalization.
