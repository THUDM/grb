import './app-home.less'
import React from 'react'
import { Button, Typography } from 'antd'
import Features from './features'
import { ReactComponent as Logo } from './logo.svg'

const { Title, Paragraph } = Typography;

const AppHomeFeature = ({icon, title, description}) => (
    <div className="app-home-feature">
      <div className="header">
        <div className="icon"><i className={`iconfont icon-${icon}`}/></div>
        <div>{title}</div>
      </div>
      <div className="content">
        {description}
      </div>
    </div>
)

export const AppHome = ({history}) => (
<div className="app-home app-container">
    <Title className="title"><Logo/></Title>
    <div className="desc">
        <Paragraph className="para">
            <b>Graph Robustness Benchmark (GRB)</b> focuses on evaluating the robustness of graph machine learning models, especially the adversarial robustness of Graph Neural Networks (GNNs).
        </Paragraph>
    </div>
    <div className="btn-group">
        <Button className="btn" type="primary" size="large" onClick={() => history.push('leaderboard')}>Go to Leaderboard</Button>
        <Button className="btn" size="large" onClick={() => history.push('docs')}>Read Documents</Button>
    </div>
    <div className="features">
        {Features.map((feature, idx) => <AppHomeFeature key={idx} {...feature}/>)}
    </div>
</div>
)