import './app-header.less'
import React from 'react'
import { Layout, Menu } from 'antd'
import { GithubOutlined } from '@ant-design/icons'
import _ from 'lodash'
import configurations from './configurations'

const { Header } = Layout;
const { SubMenu } = Menu;

export const AppHeader = ({history}) => {
    const key = _.last(history.location.pathname.split('/'))
    return <Header className="app-header" style={{ position: 'fixed', zIndex: 1, width: '100%' }}>
      <div className="header-wrapper">
        <div className="logo">Graph Robustness Benchmark</div>
        <Menu theme="dark" mode="horizontal" defaultSelectedKeys={['home']} selectedKeys={[key]} style={{float: 'right'}}>
          <Menu.Item key="home" onClick={() => history.push('/home')}>Home</Menu.Item>
          <SubMenu key="intro" title="Intro" popupOffset={[-20,-2]}>
            <Menu.Item key="get_started" onClick={() => history.push(`/intro/get_started`)}>Get Started</Menu.Item>
            <Menu.Item key="rules" onClick={() => history.push(`/intro/rules`)}>Rules</Menu.Item>
          </SubMenu>
          <Menu.Item key="datasets" onClick={() => history.push('/datasets')}>Datasets</Menu.Item>
          <SubMenu key="leaderboard" title={<span onClick={() => history.push('/leaderboard/')}>Leaderboard</span>} popupOffset={[-20,-2]}>
            {['grb-citeseer', 'grb-cora', 'grb-flickr', 'grb-reddit', 'grb-aminer'].map(dataset_name => <Menu.Item
              key={dataset_name} onClick={() => history.push(`/leaderboard/${dataset_name}`)}
            >{dataset_name}</Menu.Item>)}
          </SubMenu>
          <Menu.Item key="team" onClick={() => { window.location.href = configurations.TEAM_URL }}>Team</Menu.Item>
          <Menu.Item key="docs" onClick={() => { window.location.href = configurations.DOCS_URL }}>Docs</Menu.Item>
          <Menu.Item key="github" onClick={() => { window.location.href = configurations.GITHUB_URL }}>Github <GithubOutlined style={{marginLeft: 5, marginRight: 0}}/></Menu.Item>
        </Menu>
      </div>
    </Header>
  }