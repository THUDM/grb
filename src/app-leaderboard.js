import './app-leaderboard.less'
import React, { useState, useEffect } from 'react'
import {
    Button,
    Divider,
    Drawer,
    Table,
    Tooltip,
    Typography
} from 'antd'
import { FullscreenOutlined, FullscreenExitOutlined } from '@ant-design/icons'
import { getAttackChart, getDefenceChart, getTableColumns, getTableItems, getTableSelection, recalculateData } from './leaderboard'
import _ from 'lodash'
import { useParams } from 'react-router'
import configurations from './configurations'
import { MarkdownLoader, MarkdownPage } from './markdown-page'

const { Title, Paragraph } = Typography

export const AppLeaderboardIndex = () => {
    return <MarkdownPage url={`${configurations.GITHUB_PROXY_URL}/docs/leaderboard.md`}/>
}

export const AppLeaderboard = () => {
    const { dataset } = useParams()
    const [leaderboard, setLeaderboard] = useState({ updated_time: null, data: {}, attacks: [], models: [], difficulties: [] })
    const defaultConfigs = { difficulties: [], attacks: [], models: [], AttacksData: [], ModelsData: [], compared: undefined }
    const [configs, setConfigs] = useState(defaultConfigs)
    const [loading, setLoading] = useState(false)
    const [drawerData, setDrawerData] = useState(undefined)
    const [fullscreen, setFullscreen] = useState(false)
    useEffect(() => {
        setConfigs(defaultConfigs)
        setLoading(true)
        Promise.all([
            fetch(`${configurations.GITHUB_PROXY_URL}/results/leaderboards/${dataset.split('-')[1]}.json`).then(resp => resp.json()),
            fetch(`${configurations.GITHUB_PROXY_URL}/results/meta/attacks.json`).then(resp => resp.json()),
            fetch(`${configurations.GITHUB_PROXY_URL}/results/meta/models.json`).then(resp => resp.json()),
        ]).then(data => {
            let lb = data[0], AttacksData = data[1], ModelsData = data[2]
            lb.attacks = _.filter(lb.attacks, x => x !== 'no_attack')
            setLeaderboard(lb)
            setConfigs({
                difficulties: ['full'],
                attacks: _.clone(lb.attacks.slice(0, 5)),
                models: _.clone(lb.models.slice(0, 10)),
                AttacksData, ModelsData
            })
            setLoading(false)
        })
    }, [dataset])
    const columns = getTableColumns(configs, setDrawerData, setConfigs, leaderboard.data)
    const data = recalculateData(leaderboard.data, configs, leaderboard.difficulties)
    const items = getTableItems(data, configs)
    return <div className="app-leaderboard app-container" style={{ width: '100%', paddingTop: 30, paddingBottom: 30 }}>
        <Title style={{ textAlign: 'center' }}><i>{dataset.toLowerCase()}</i> Challenge</Title>
        <Divider style={{ marginTop: 50, fontSize: 24 }}>
            Leaderboard
            <sup><a style={{marginLeft: 5}} onClick={() => setFullscreen(true)}><FullscreenOutlined /></a></sup>
        </Divider>
        <Table loading={loading} columns={columns} dataSource={items} bordered pagination={false} scroll={{ y: '80vh' }}/>
        {fullscreen && <div className="popup fullscreen">
            <div className="holder" onClick={() => setFullscreen(false)}/>
            <div className="exit-fullscreen"onClick={() => setFullscreen(false)}><Button shape="circle" type="primary" size="large" icon={<FullscreenExitOutlined />}/></div>
            <Table loading={loading} columns={columns} dataSource={items} bordered pagination={false} scroll={{ x: '90vw', y: '80vh' }}/>
        </div>}
        <div className="notes">
            <MarkdownLoader url={`${configurations.GITHUB_PROXY_URL}/docs/leaderboard-hint.md`}/>
        </div>
        <Divider style={{ marginTop: 50, fontSize: 24 }}>Configurations</Divider>
        {getTableSelection('attacks', leaderboard, configs, setConfigs)}
        {getTableSelection('models', leaderboard, configs, setConfigs)}
        {getTableSelection('difficulties', leaderboard, configs, setConfigs)}
        <div style={{ marginTop: 10, marginBottom: 10, display: "flex", justifyContent: "center" }}>
            <Tooltip title="The best 3 attacks and best 5 defense models">
                <Button style={{ width: 150, margin: 10 }} onClick={() => setConfigs({ ...configs, difficulties: ['full'], models: leaderboard.models.slice(0, 5), attacks: leaderboard.attacks.slice(0, 3) })}>Brief</Button>
            </Tooltip>
            <Tooltip title="The best 5 attacks and best 10 defense models">
                <Button style={{ width: 150, margin: 10 }} onClick={() => setConfigs({ ...configs, difficulties: _.clone(leaderboard.difficulties), models: leaderboard.models.slice(0, 10), attacks: leaderboard.attacks.slice(0, 5) })}>Main</Button>
            </Tooltip>
            <Tooltip title="All attacks, all defense models under all difficulties.">
                <Button style={{ width: 150, margin: 10 }} onClick={() => setConfigs({ ...configs, difficulties: _.clone(leaderboard.difficulties), models: _.clone(leaderboard.models), attacks: _.clone(leaderboard.attacks) })}>Completed</Button>
            </Tooltip>
            <Tooltip title="Only raw models without any defense mechanism.">
                <Button style={{ width: 150, margin: 10 }} onClick={() => setConfigs({ ...configs, models: _.clone(leaderboard.models.filter(x => x.split('_').length === 1 && x.indexOf('guard') === -1 && x.indexOf('robust') === -1 && x.indexOf('svd') === -1)) })}>No Defense</Button>
            </Tooltip>
        </div>
        <Divider style={{ marginTop: 50, fontSize: 24 }}>Ranking Charts of Attacks and Defenses</Divider>
        <div className="charts" style={{ width: '100%' }}>
            {[
                getAttackChart(data, leaderboard.difficulties, 'weighted', configs),
                getDefenceChart(data, leaderboard.difficulties, 'weighted', configs)
            ].map((chart, idx) =>
                <div key={idx} className="chart" style={{ width: 'calc(50% - 20px)', margin: 10, display: 'inline-block', height: `${90 * Math.max(configs.models.length, configs.attacks.length)}px` }}>
                    <div style={{ textAlign: "center", fontSize: 16, fontWeight: 500, marginBottom: 10, marginTop: 20 }}>{(idx === 0) ? 'Attacks' : 'Defenses'}</div>
                    {/* {getChart(leaderboard.data, difficulty, configs)} */}
                    {chart}
                </div>
            )}
        </div>
        <Drawer width="360px" title={drawerData ? drawerData.title : ''} placement="right" closable={false} onClose={() => setDrawerData(undefined)} visible={drawerData !== undefined}>
            {drawerData && drawerData.description}
        </Drawer>
    </div>
}
