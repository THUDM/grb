import React from 'react';
import { Badge, Select, Typography, Tooltip } from "antd";
import _ from "lodash";
import { Bar } from '@ant-design/charts';
import { std, sum, min, max } from 'mathjs';

const { Option } = Select;
const { Title, Paragraph } = Typography;

function renderSummaryCell(summary, type) {
    if (type === 'model') {
        if (summary === 'average') return 'Average Accuracy'
        else if (summary === '3-max') return 'Average 3-Max Accuracy'
        else if (summary === 'weighted') return 'Weighted Accuracy'
        else return summary
    } else {
        if (summary === 'no_attack') return 'W/O Attack'
        else if (summary === 'average') return 'Average Accuracy'
        else if (summary === '3-min') return 'Average 3-Min Accuracy'
        else if (summary === 'weighted') return 'Weighted Accuracy'
        else return summary
    }
}

function renderDrawer(title, refs, desc) {
    return {
        'title': <Title level={4}>{title}</Title>,
        'description': <div className="description">
            {desc && <Title level={5}>Description</Title>}
            {desc && <Paragraph>{desc}</Paragraph>}
            <Title level={5}>References</Title>
            {refs.map((ref, idx) => <Paragraph key={idx}>[{idx+1}] {ref}</Paragraph>)}
        </div>
    }
}

function renderModelHeader(model_id, updateDrawer, ModelsData) {
    const model = ModelsData.find(model => model.id === model_id)
    const layer_norm = model_id.split('_').indexOf('ln') >= 0
    const adversarial_training = model_id.split('_').indexOf('at') >= 0
    const inner = <a onClick={() => updateDrawer(convertModelDataToDrawerData(model_id, ModelsData))}>{model.name}</a>
    return <div>
        {inner}
        {layer_norm && <Tooltip title="Layer Normalization"><sub className="model-header-badge" style={{color: '#1890ff'}}>+LN</sub></Tooltip>}
        {adversarial_training && <Tooltip title="Adversarial Training"><sub className="model-header-badge" style={{color: '#ff9c6e'}}>+AT</sub></Tooltip>}
    </div>
}

function convertAttackDataToDrawerData(attack_id, AttacksData) {
    const atk = AttacksData.find(atk => atk.id === attack_id)
    if (!atk) return undefined
    let refs = atk.refs || []
    if (atk.ref) refs.push(atk.ref)
    return renderDrawer(atk.id.toUpperCase(), refs, atk.desc || '')
}

function convertModelDataToDrawerData(model_id, ModelsData) {
    const model = ModelsData.find(model => model.id === model_id)
    if (!model) return undefined
    let refs = model.refs || []
    if (model.ref) refs.push(model.ref)
    return renderDrawer(model.id.split('_')[0].toUpperCase(), refs, model.desc || '')
}

export function getTableColumns(configs, updateDrawer) {
    const {difficulties, models, AttacksData, ModelsData} = configs
    const width = configs.width || 120
    return [{
        title: "Rank",
        fixed: 'left',
        align: 'center',
        width: 60,
        render: (value, row, index) => {
            if (!row.isFirstRow) return {props: {rowSpan: 0, colSpan: 0}}
            else {
                if (row.rank === 0) return {children: <b>{renderSummaryCell(row.attack, 'attack')}</b>, props: {rowSpan: difficulties.length, colSpan: 2}}
                else return {children: <b>{row.rank}</b>, props: {rowSpan: difficulties.length, colSpan: 1}}
            }
        }
    }, {
        title: "Attack",
        fixed: 'left',
        align: 'center',
        width: 75,
        render: (value, row, index) => {
            if (!row.isFirstRow) return {props: {rowSpan: 0, colSpan: 0}}
            else {
                if (row.rank === 0) return {props: {rowSpan: 0, colSpan: 0}}
                else return {
                    children: <a className="table-attack-header" onClick={() => updateDrawer(convertAttackDataToDrawerData(row.attack, AttacksData))}>
                        <b>{row.attack.toUpperCase()}</b>
                    </a>,
                    props: {rowSpan: difficulties.length, colSpan: 1}}
            }
        }
    }, {
        title: "Difficulty",
        fixed: 'left',
        align: 'center',
        width: 80,
        render: (value, row, index) => {
            return {children: <b>{_.capitalize(row.difficulty)}</b>, props: {rowSpan: 1, colSpan: 1}}
        }
    }, {
        title: "Models",
        children: models.map(model => {
            return {
                title: renderModelHeader(model, updateDrawer, ModelsData),
                dataIndex: model,
                key: model,
                align: 'center',
                width,
                render: (value, row, index) => {
                    const inner = <span className="value">{value.mean ? value.mean.toFixed(2) : '-'}{(value.std !== null) && <sub className="std">±{value.std.toFixed(2)}</sub>}</span>
                    return row.bold.indexOf(model) >= 0 ? <b>{inner}</b> : inner
                }
            }
        })
    }].concat(['average', '3-max', 'weighted'].map(model => {
        return {
            title: renderSummaryCell(model, 'model'),
            dataIndex: model,
            key: model,
            align: 'center',
            width,
            render: (value, row, index) => {
                if (value === undefined) return '-'
                const inner = <span className="value">{value.mean ? value.mean.toFixed(2) : '-'}{(value.std !== null) && <sub className="std">±{value.std.toFixed(2)}</sub>}</span>
                return row.bold.indexOf(model) >= 0 ? <b>{inner}</b> : inner
            }
        }
    }))
}

function weightedFunc(arr) {
    return sum(arr.map((v, i) => v / (i + 1) / (i + 1))) / sum(arr.map((v, i) => 1 / (i + 1) / (i + 1)))
}

export function recalculateData(data, configs, difficulties) {
    const {attacks, models} = configs
    difficulties.forEach(difficulty => {
        // calculate modelsSummary
        attacks.forEach(atk => {
            const modelScores = models.map(m => data[atk][difficulty][m])
            const size = min(modelScores.map(s => s.values.length))
            let averageValues = []
            let threeMaxValues = []
            let weightedValues = []
            _.range(size).forEach(i => {
                const _modelValues = modelScores.map(s => s.values[i])
                _modelValues.sort((a, b) => b - a)
                averageValues.push(_.mean(_modelValues))
                threeMaxValues.push(_.mean(_modelValues.slice(0, 3)))
                weightedValues.push(weightedFunc(_modelValues))
            });
            [['average', averageValues], ['3-max', threeMaxValues], ['weighted', weightedValues]].forEach(tup => {
                const m = tup[0]
                const values = tup[1]
                if (!data[atk]) data[atk] = {}
                if (!data[atk][difficulty]) data[atk][difficulty] = {}
                data[atk][difficulty][m] = {
                    'values': values,
                    'mean': _.mean(values),
                    'std': std(values)
                }
            })
        })
        // calculate attacksSummary
        models.forEach(m => {
            const attackScores = attacks.map(atk => data[atk][difficulty][m])
            const size = min(attackScores.map(s => s.values.length))
            let averageValues = []
            let threeMinValues = []
            let weightedValues = []
            _.range(size).forEach(i => {
                const _attackValues = attackScores.map(s => s.values[i])
                _attackValues.sort((a, b) => a - b)
                averageValues.push(_.mean(_attackValues))
                threeMinValues.push(_.mean(_attackValues.slice(0, 3)))
                weightedValues.push(weightedFunc(_attackValues))
            });
            [['average', averageValues], ['3-min', threeMinValues], ['weighted', weightedValues]].forEach(tup => {
                const atk = tup[0]
                const values = tup[1]
                if (!data[atk]) data[atk] = {}
                if (!data[atk][difficulty]) data[atk][difficulty] = {}
                data[atk][difficulty][m] = {
                    'values': values,
                    'mean': _.mean(values),
                    'std': std(values)
                }
            })
        })
    })
    return data
}

export function getTableItems(data, configs) {
    const {attacks, models, difficulties} = configs
    const attacksSummary = ['no_attack', 'average', '3-min', 'weighted']
    const modelsSummary = ['average', '3-max', 'weighted']
    const bestModelScore = {}
    attacksSummary.forEach(attackSummary => difficulties.forEach(difficulty => {
        let best = 0
        models.forEach(model => best = Math.max(best, data[attackSummary][difficulty][model].mean))
        if (!bestModelScore[attackSummary]) bestModelScore[attackSummary] = {}
        bestModelScore[attackSummary][difficulty] = best
    }))
    const bestAttackScore = {}
    difficulties.forEach(difficulty => modelsSummary.forEach(modelSummary => {
        let best = 100
        attacks.forEach(attack => best = Math.min(best, data[attack][difficulty][modelSummary].mean))
        if (!bestAttackScore[difficulty]) bestAttackScore[difficulty] = {}
        bestAttackScore[difficulty][modelSummary] = best
    }))

    const items = _.flatMap(attacks.concat(attacksSummary), atk => {
        return difficulties.map((difficulty, lid) => {
            let item = {
                'attack': atk,
                'rank': attacks.indexOf(atk) + 1,
                'difficulty': difficulty,
                'isFirstRow': lid === 0,
                'bold': []
            }
            models.concat(modelsSummary).forEach(m => {
                item[m] = data[atk][difficulty][m]
                if (attacksSummary.indexOf(atk) >= 0 && models.indexOf(m) >= 0) {
                    if (data[atk][difficulty][m].mean === bestModelScore[atk][difficulty]) item['bold'].push(m)
                } else if (attacks.indexOf(atk) >= 0 && modelsSummary.indexOf(m) >= 0) {
                    if (data[atk][difficulty][m].mean === bestAttackScore[difficulty][m]) item['bold'].push(m)
                }
            })
            return item
        })
    })
    return items
}

export function getTableSelection(key, leaderboard, configs, setConfigs) {
    return <div className="table-config">
        <div style={{width: 100}}><Title level={5}>{(key === 'difficulties') ? 'Difficulty' : _.capitalize(key)}:</Title></div>
        <div className="selection-box">
        {leaderboard[key] && <Select mode="multiple" value={configs[key]} onChange={(value) => {
            const newConfigs = _.clone(configs)
            newConfigs[key] = value.sort((a, b) => leaderboard[key].indexOf(a) - leaderboard[key].indexOf(b))
            setConfigs(newConfigs)
        }} style={{width: '100%', marginLeft: 10}}>
            {leaderboard[key].map(v => <Option key={v}>{
                (key === 'attacks') ? v.toUpperCase() :
                (key === 'difficulties') ? _.capitalize(v) : v}</Option>)}
        </Select>}
        </div>
    </div>
}

export function getAttackChart(data, difficulties, summary, configs) {
    const {attacks} = configs
    let barData = _.flatMap(attacks, atk => difficulties.map(difficulty => {
        return { label: atk.toUpperCase(), type: _.capitalize(difficulty), value: parseFloat(data[atk][difficulty][summary].mean.toFixed(2)) }
    }))
    return <Bar data={barData} isGroup xField="value" yField="label" seriesField="type" marginRatio={0.1}
        label={{
            position: "middle",
            layout: [
                { type: 'interval-adjust-position' },
                { type: 'adjust-color' },
            ]
    }}/>
}

export function getDefenceChart(data, difficulties, summary, configs) {
    const {models} = configs
    let barData = _.flatMap(models, model => difficulties.map(difficulty => {
        return { label: model, type: _.capitalize(difficulty), value: parseFloat(data[summary][difficulty][model].mean.toFixed(2)) }
    }))
    return <Bar data={barData} isGroup xField="value" yField="label" seriesField="type" marginRatio={0.1}
        label={{
            position: "middle",
            layout: [
                { type: 'interval-adjust-position' },
                { type: 'adjust-color' },
            ]
    }}/>
}

export function getChart(data, difficulty, configs) {
    const {attacks, modelsSummary} = configs
    let barData = _.flatMap(attacks, atk => modelsSummary.map(modelSummary => {
        return { label: atk.toUpperCase(), type: modelSummary, value: parseFloat(data[atk][difficulty][modelSummary].mean.toFixed(2)) }
    }))
    return <Bar data={barData} isGroup xField="value" yField="label" seriesField="type" marginRatio={0.1}
        label={{
            position: "middle",
            layout: [
                { type: 'interval-adjust-position' },
                { type: 'interval-hide-overlap' },
                { type: 'adjust-color' },
            ]
    }}/>
}