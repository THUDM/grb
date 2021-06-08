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
import { getAttackChart, getDefenceChart, getTableColumns, getTableItems, getTableSelection, recalculateData } from './leaderboard'
import _ from 'lodash'
import { useParams } from 'react-router'

const { Title } = Typography

export const AppLeaderboard = () => {
    const { dataset } = useParams()
    const [leaderboard, setLeaderboard] = useState({ updated_time: null, data: {}, attacks: [], models: [], difficulties: [] })
    const [configs, setConfigs] = useState({ difficulties: [], attacks: [], models: [] })
    const [loading, setLoading] = useState(false)
    const [drawerData, setDrawerData] = useState(undefined)
    useEffect(() => {
        setLoading(true)
        fetch(`${process.env.PUBLIC_URL}/leaderboards/${dataset}.json`)
            .then(resp => resp.json()).then(lb => {
                setLeaderboard(lb)
                setConfigs({
                    difficulties: ['full'], //_.clone(lb.difficulties),
                    attacks: _.clone(lb.attacks.slice(0, 5)),
                    models: _.clone(lb.models.slice(0, 10))
                })
                setLoading(false)
            })
    }, [dataset])
    const columns = getTableColumns(configs, setDrawerData)
    const data = recalculateData(leaderboard.data, configs, leaderboard.difficulties)
    const items = getTableItems(data, configs)
    return <div className="app-leaderboard app-container" style={{ width: '100%', paddingTop: 30, paddingBottom: 30 }}>
        <Title style={{ textAlign: 'center' }}>{_.capitalize(dataset)} Challenge</Title>
        <Divider style={{ marginTop: 50, fontSize: 24 }}>Leaderboard</Divider>
        <Table loading={loading} columns={columns} dataSource={items} bordered pagination={false} scroll={{ x: 1300, y: '80vh' }} />
        <Divider style={{ marginTop: 50, fontSize: 24 }}>Configurations</Divider>
        {getTableSelection('attacks', leaderboard, configs, setConfigs)}
        {getTableSelection('models', leaderboard, configs, setConfigs)}
        {getTableSelection('difficulties', leaderboard, configs, setConfigs)}
        <div style={{ marginTop: 10, marginBottom: 10, display: "flex", justifyContent: "center" }}>
            <Tooltip title="The best 3 attacks and best 5 defence models">
                <Button style={{ width: 150, margin: 10 }} onClick={() => setConfigs({ ...configs, difficulties: ['full'], models: leaderboard.models.slice(0, 5), attacks: leaderboard.attacks.slice(0, 3) })}>Brief</Button>
            </Tooltip>
            <Tooltip title="The best 5 attacks and best 10 defence models">
                <Button style={{ width: 150, margin: 10 }} onClick={() => setConfigs({ ...configs, difficulties: ['full'], models: leaderboard.models.slice(0, 10), attacks: leaderboard.attacks.slice(0, 5) })}>Main</Button>
            </Tooltip>
            <Tooltip title="All attacks, all defence models, and all difficulties of attacks.">
                <Button style={{ width: 150, margin: 10 }} onClick={() => setConfigs({ ...configs, difficulties: _.clone(leaderboard.difficulties), models: _.clone(leaderboard.models), attacks: _.clone(leaderboard.attacks) })}>Completed</Button>
            </Tooltip>
            <Tooltip title="Only raw models without any defence (layer normalization and adversarial training)">
                <Button style={{ width: 150, margin: 10 }} onClick={() => setConfigs({ ...configs, models: _.clone(leaderboard.models.filter(x => x.split('_').length === 1 && x.indexOf('guard') === -1 && x.indexOf('robust') === -1 && x.indexOf('svd') === -1)) })}>No Defence</Button>
            </Tooltip>
        </div>
        <Divider style={{ marginTop: 50, fontSize: 24 }}>Comparison of Attacks and Defences</Divider>
        <div className="charts" style={{ width: '100%' }}>
            {[
                getAttackChart(data, leaderboard.difficulties, 'weighted', configs),
                getDefenceChart(data, leaderboard.difficulties, 'weighted', configs)
            ].map((chart, idx) =>
                <div key={idx} className="chart" style={{ width: 'calc(50% - 20px)', margin: 10, display: 'inline-block', height: `${90 * Math.max(configs.models.length, configs.attacks.length)}px` }}>
                    <div style={{ textAlign: "center", fontSize: 16, fontWeight: 500, marginBottom: 10, marginTop: 20 }}>{(idx === 0) ? 'Attack' : 'Defence'}</div>
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
