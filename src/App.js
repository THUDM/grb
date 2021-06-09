import React from 'react';
import { Layout } from 'antd';
import { BrowserRouter as Router, Switch, Route, Redirect } from 'react-router-dom';
import './App.less';

import { AppDatasets } from './app-datasets';
import { AppDocs } from './app-docs'
import { AppHome } from './app-home';
import { AppLeaderboard } from './app-leaderboard';
import { AppHeader } from './app-header';
import { AppIntro } from './app-intro';

const { Content, Footer } = Layout;

const App = () => (
  <Router basename="/grb/">
    <Layout>
      <Switch>
        <Route path="*" render={({history}) => <AppHeader history={history}/>}/>
      </Switch>
      <Content className="site-layout">
        <Switch>
          <Route path="/home" render={({history}) => <AppHome history={history}/>}/>
          <Route path="/intro/:entry" render={() => <AppIntro/>}/>
          <Route path="/docs" render={() => <AppDocs/>}/>
          <Route path="/datasets" render={({history}) => <AppDatasets history={history}/>}/>
          <Route path="/leaderboard/:dataset" render={({history}) => <AppLeaderboard history={history}/>}/>
          <Route path="*"><Redirect to="/home"/></Route>
        </Switch>
      </Content>
      <Footer className="app-footer">Knowledge Engineering Group, Department of Computer Science and Technology, Tsinghua University</Footer>
    </Layout>
  </Router>
);

export default App;