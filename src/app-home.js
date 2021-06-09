import './app-home.less'
import React from 'react'
import { Button, Typography } from 'antd'
import configurations from './configurations'
import { ReactComponent as Logo } from './logo.svg'

const { Title, Paragraph } = Typography;
const { Features } = configurations;

const AppHomeFeature = ({icon, title, description}) => (
    <div className="app-home-feature">
      <div className="header">
        <div className="icon"><i className={`iconfont icon-${icon}`}/></div>
        <div className="title">{title}</div>
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
          <b>Graph Robustness Benchmark (GRB)</b> provides scalable, general, unified, and reproducible evaluation on the adversarial robustness of graph machine learning, especially Graph Neural Networks (GNNs). GRB has elaborated datasets, unified evaluation pipeline, reproducible leaderboards, and modular coding framework, which facilitates a fair comparison among various attacks & defenses on GNNs and promotes future research in this field.
        </Paragraph>
    </div>
    <div className="btn-group">
        <Button className="btn" type="primary" size="large" onClick={() => history.push('/intro/get_started')}>Quick Start</Button>
        <Button className="btn" size="large" onClick={() => history.push('docs')}>Read Documents</Button>
    </div>
    <div className="features">
        {Features.map((feature, idx) => <AppHomeFeature key={idx} {...feature}/>)}
    </div>
</div>
)