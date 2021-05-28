import React, { useState, useEffect } from 'react';
import {
  Button,
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

const AppHome = ({history}) => (
  <div className="home" style={{maxWidth: 720}}>
    <Title style={{textAlign: 'center'}}>
      {/* Graph Robustness Benchmark */}
      <Logo/>
    </Title>
    <Paragraph style={{fontSize: 24, marginTop: 30, marginBottom: 30}}>
      <b>Graph Robustness Benchmark (GRB)</b> focuses on evaluating the robustness of graph machine learning models, especially the adversarial robustness of Graph Neural Networks (GNNs).
    </Paragraph>
    <div style={{display: "flex", justifyContent: "center"}}>
      <Button style={{margin: 10, width: 180}} type="primary" size="large" onClick={() => history.push('leaderboard')}>Go to Leaderboard</Button>
      <Button style={{margin: 10, width: 180}} size="large" onClick={() => history.push('docs')}>Read Documents</Button>
    </div>
  </div>
)

const AppLeaderboard = () => {
  const [data, setData] = useState({datagrid: [], methods: [], models: [], metrics: [], columns: [], items: []})
  const [loading, setLoading] = useState(false)
  useEffect(() => {
    setLoading(true)
    // fetch(`http://webplus-cn-zhangjiakou-s-5d3021e74130ed2505537ee6.oss-cn-zhangjiakou.aliyuncs.com/keg/grb.csv`)
    fetch(`https://raw.githubusercontent.com/Stanislas0/grb/results/leaderboard.csv?token=ADK6VYPSLY6CFL6INBUZKGLAXJFW4`)
      .then(resp => resp.text()).then(text => {
        const datagrid = text.split('\n').map(line => line.split(';'))
        const nrows = datagrid.length, ncols = datagrid[0].length;
        let models = [], metrics = [];
        for (let i = 3; i < ncols; ++i) {
          if (datagrid[1][i].length === 0) metrics.push({colIndex: i, name: datagrid[0][i]})
          else models.push({colIndex: i, id: parseInt(datagrid[0][i]), name: datagrid[1][i]})
        }
        let methods = [], currentMethod = {};
        for (let i = 2; i < nrows; ++i) {
          if (datagrid[i][0].length > 0) {
            if (currentMethod.type) methods.push(currentMethod)
            if (datagrid[i][1].length > 0) currentMethod = {type: "method", rank: parseInt(datagrid[i][0]), name: datagrid[i][1], subKeys: [{rowIndex: i, name: datagrid[i][2]}]}
            else currentMethod = {rowIndex: i, type: "summary", name: datagrid[i][0]}
          } else {
            if (currentMethod.subKeys) currentMethod.subKeys.push({rowIndex: i, name: datagrid[i][2]})
          }
        }
        if (currentMethod.type) methods.push(currentMethod)
        const columns = [
          {title: "Rank", colSpan: 3, fixed: 'left', align: 'center', render: (value, row, index) => {
            if (row.method.length > 0 && row.rank.length === 0) return {children: row.method, props: {colSpan: 3}}
            else if (row.rowSpan !== 1) return {children: row.rank, props: {rowSpan: row.rowSpan}}
            else return {children: row.rank, props: {colSpan: 1}}
          }},
          {title: "Method", colSpan: 0, fixed: 'left', align: 'center', render: (value, row, index) => {
            if (row.method.length > 0 && row.rank.length === 0) return {children: "", props: {colSpan: 0}}
            else if (row.rowSpan !== 1) return {children: row.method, props: {rowSpan: row.rowSpan}}
            else return {children: row.method, props: {colSpan: 1}}
          }},
          {title: "SubKey", colSpan: 0, fixed: 'left', align: 'center', render: (value, row, index) => {
            if (row.method.length > 0 && row.rank.length === 0) return {children: "", props: {colSpan: 0}}
            else return {children: row.subKey, props: {colSpan: 1}}
          }},
          {title: "Models", children: models.map(model => { return {title: model.name, dataIndex: model.name, key: model.name, align: 'center'} })}
        ].concat(metrics.map(metric => { return {title: metric.name, dataIndex: metric.name, key: metric.name, align: 'center'} }))

        const parseValue = (value) => {
          try {
            const s = parseFloat(value).toFixed(4);
            return (s === 'NaN') ? '-' : s;
          } catch {
            return value;
          }
        }
        let items = _.flatten(methods.map(method => {
          if (method.type === "method") return method.subKeys.map((key, idx) => {
            let item = {key: `${method.name}.${key.name}`, rank: idx === 0 ? method.rank : "", method: idx === 0 ? method.name : "", subKey: key.name, rowSpan: idx === 0 ? method.subKeys.length : 0}
            models.forEach(model => item[model.name] = parseValue(datagrid[key.rowIndex][model.colIndex]))
            metrics.forEach(metric => item[metric.name] = parseValue(datagrid[key.rowIndex][metric.colIndex]))
            return item
          })
          else {
            let item = {key: method.name, rank: "", method: method.name, subKey: "", rowSpan: 1}
            models.forEach(model => item[model.name] = parseValue(datagrid[method.rowIndex][model.colIndex]))
            metrics.forEach(metric => item[metric.name] = parseValue(datagrid[method.rowIndex][metric.colIndex]))
            return [item]
          }
        }))
        console.log({datagrid, methods, models, metrics, columns, items})
        setLoading(false)
        setData({datagrid, methods, models, metrics, columns, items})
      })
  }, [])
  return <div className="leaderboard" style={{width: '100%'}}>
    <Title style={{textAlign: 'center'}}>Leaderboard</Title>
    <Table loading={loading} columns={data.columns} dataSource={data.items} bordered pagination={false} scroll={{y: '60vh'}}/>
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
          <Route path="/docs"></Route>
          <Route path="/datasets"></Route>
          <Route path="/leaderboard" render={({history}) => <AppLeaderboard history={history}/>}/>
          <Route path="*"><Redirect to="/home"/></Route>
        </Switch>
        {/* <Breadcrumb style={{ margin: '16px 0' }}>
          <Breadcrumb.Item>Home</Breadcrumb.Item>
          <Breadcrumb.Item>List</Breadcrumb.Item>
          <Breadcrumb.Item>App</Breadcrumb.Item>
        </Breadcrumb>
        <div className="site-layout-background" style={{ padding: 24, minHeight: 380 }}>
          Content
        </div> */}
      </Content>
      <Footer style={{ textAlign: 'center' }}>Knowledge Engineering Group, Department of Computer Science and Technology, Tsinghua University</Footer>
    </Layout>
  </Router>
);

export default App;