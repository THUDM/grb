import React, { useState, useEffect } from 'react';
import {
  Button,
  Divider,
  Drawer,
  Layout,
  Menu,
  Spin,
  Table,
  Typography,
} from 'antd';
import { GithubOutlined } from '@ant-design/icons'
import { BrowserRouter as Router, Switch, Route, Redirect } from 'react-router-dom';
import './App.less';
import _ from 'lodash';
import { ReactComponent as Logo } from './logo.svg';
import Features from './features';
import ReactMarkdown from 'react-markdown';
import { getChart, getTableColumns, getTableItems, getTableSelection } from './leaderboard';

const { Header, Content, Footer } = Layout;
const { Title, Paragraph } = Typography;

const AppHeader = ({history}) => {
  const key = history.location.pathname.split('/')[1]
  return <Header style={{ position: 'fixed', zIndex: 1, width: '100%' }}>
    <div className="logo">Graph Robustness Benchmark</div>
    <Menu theme="dark" mode="horizontal" defaultSelectedKeys={['home']} selectedKeys={[key]} style={{float: 'right'}}>
      <Menu.Item key="home" onClick={() => history.push('/home')}>Home</Menu.Item>
      <Menu.Item key="docs" onClick={() => history.push('/docs')}>Docs</Menu.Item>
      <Menu.Item key="datasets" onClick={() => history.push('/datasets')}>Datasets</Menu.Item>
      <Menu.Item key="leaderboard" onClick={() => history.push('/leaderboard')}>Leaderboard</Menu.Item>
      <Menu.Item key="team" onClick={() => { window.location.href = 'http://keg.cs.tsinghua.edu.cn/' }}>Team</Menu.Item>
      <Menu.Item key="github" onClick={() => { window.location.href = 'https://github.com/Stanislas0/grb' }}>Github <GithubOutlined style={{marginLeft: 5, marginRight: 0}}/></Menu.Item>
    </Menu>
  </Header>
}

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

const AppHome = ({history}) => (
  <div className="home">
    <Title style={{textAlign: 'center'}}>
      {/* Graph Robustness Benchmark */}
      <Logo/>
    </Title>
    <div className="desc">
      <Paragraph style={{fontSize: '1.35em', marginTop: 30, marginBottom: 30, maxWidth: 720}}>
        <b>Graph Robustness Benchmark (GRB)</b> focuses on evaluating the robustness of graph machine learning models, especially the adversarial robustness of Graph Neural Networks (GNNs).
      </Paragraph>
    </div>
    <div style={{display: "flex", justifyContent: "center"}}>
      <Button style={{margin: 10, width: 180}} type="primary" size="large" onClick={() => history.push('leaderboard')}>Go to Leaderboard</Button>
      <Button style={{margin: 10, width: 180}} size="large" onClick={() => history.push('docs')}>Read Documents</Button>
    </div>
    <div className="features">
      {Features.map((feature, idx) => <AppHomeFeature key={idx} {...feature}/>)}
    </div>
  </div>
)

const AppDoc = () => {
  const [data, setData] = useState("")
  const [loading, setLoading] = useState(false)
  useEffect(() => {
    setLoading(true)
    fetch('https://raw.githubusercontent.com/Stanislas0/grb/main/README.md?token=ADK6VYNK27NGG7I3IQR53QDAXM7CS')
      .then(resp => resp.text()).then(text => {
        setLoading(false)
        setData(text)
      })
  }, [])
  return <div className="docs">
    <Spin spinning={loading}>
      <ReactMarkdown>{data}</ReactMarkdown>
    </Spin>
  </div>
}

const AppLeaderboard = () => {
  const [leaderboard, setLeaderboard] = useState({updated_time: null, data: {}, attacks: [], models: [], levels: [], attacksSummary: [], modelsSummary: []})
  const [configs, setConfigs] = useState({levels: [], attacks: [], models: [], attacksSummary: [], modelsSummary: []})
  const [loading, setLoading] = useState(false)
  const [drawerData, setDrawerData] = useState(undefined)
  useEffect(() => {
    setLoading(true)
    fetch(`https://raw.githubusercontent.com/Stanislas0/grb/results/leaderboard.json?token=ADK6VYJ3MU26KVKUN3DS6HDAYL2DC`)
        .then(resp => resp.json()).then(lb => {
          console.log('lb', lb)
          setLeaderboard(lb)
          setConfigs({
            levels: _.clone(lb.levels),
            attacks: _.clone(lb.attacks),
            models: _.clone(lb.models),
            attacksSummary: _.clone(lb.attacks_summary),
            modelsSummary: _.clone(lb.models_summary)
          })
          setLoading(false)
        })
  }, [])
  const columns = getTableColumns(configs, setDrawerData)
  const items = getTableItems(leaderboard.data, configs)
  return <div className="leaderboard" style={{width: '100%', paddingTop: 30, paddingBottom: 30}}>
    <Title style={{textAlign: 'center'}}>Leaderboard</Title>
    {getTableSelection('attacks', leaderboard, configs, setConfigs)}
    {getTableSelection('models', leaderboard, configs, setConfigs)}
    {getTableSelection('levels', leaderboard, configs, setConfigs)}
    <div style={{marginTop: 10, marginBottom: 10, display: "flex", justifyContent: "center"}}>
      <Button style={{width: 150, margin: 10}} onClick={() => setConfigs({...configs, levels: ['full'], models: leaderboard.models.slice(0, 5), attacks: leaderboard.attacks.slice(0, 3)})}>Brief</Button>
      <Button style={{width: 150, margin: 10}} onClick={() => setConfigs({...configs, levels: _.clone(leaderboard.levels), models: _.clone(leaderboard.models), attacks: _.clone(leaderboard.attacks)})}>Completed</Button>
    </div>
    <Table loading={loading} columns={columns} dataSource={items} bordered pagination={false} scroll={{x: 1300, y: '80vh'}}/>
    <Divider style={{marginTop: 50, fontSize: 24}}>Comparison of Attacks</Divider>
    <div className="charts" style={{width: '100%'}}>
      {leaderboard.levels.map(level =>
        <div className="chart" style={{width: 'calc(50% - 20px)', margin: 10, display: 'inline-block'}}>
          <div id={`level-chart:${level}`} style={{textAlign: "center", fontSize: 16, fontWeight: 500, marginBottom: 10, marginTop: 20}}>{_.capitalize(level)}</div>
          {getChart(leaderboard.data, level, configs)}
        </div>
      )}
    </div>
    <Drawer width="360px" title={drawerData ? drawerData.title : ''} placement="right" closable={false} onClose={() => setDrawerData(undefined)} visible={drawerData !== undefined}>
      {drawerData && drawerData.description}
    </Drawer>
  </div>
}

const App = () => (
  <Router basename="/grb/">
    <Layout>
      <Switch>
        <Route path="*" render={({history}) => <AppHeader history={history}/>}/>
      </Switch>
      <Content className="site-layout" style={{ padding: '0 50px', marginTop: 64, minHeight: 'calc(100vh - 134px)', display: "flex", justifyContent: "center", alignItems: "center" }}>
        <Switch>
          <Route path="/home" render={({history}) => <AppHome history={history}/>}/>
          <Route path="/docs" render={() => <AppDoc/>}/>
          <Route path="/datasets"></Route>
          <Route path="/leaderboard" render={({history}) => <AppLeaderboard history={history}/>}/>
          <Route path="*"><Redirect to="/home"/></Route>
        </Switch>
      </Content>
      <Footer style={{ textAlign: 'center', color: '#888' }}>Knowledge Engineering Group, Department of Computer Science and Technology, Tsinghua University</Footer>
    </Layout>
  </Router>
);

export default App;