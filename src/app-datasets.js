import './app-datasets.less'
import { Divider, Table, Tooltip, Typography } from 'antd'
import _ from 'lodash';
import React, { useEffect, useState } from 'react'
import { BarChartOutlined, FilePdfOutlined, CloudDownloadOutlined } from '@ant-design/icons'
import configurations from './configurations';

const { Title, Paragraph } = Typography;

function numberWithCommas(x) {
    return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function getDatasetTableColumns() {
    return [{
        title: 'Name',
        fixed: 'left',
        align: 'center',
        width: 80,
        render: (value, row, index) => {
            return {children: <a href={`#${row.name}`}><b><i>grb-{row.name.toLowerCase()}</i></b></a>}
        }
    }, {
        title: 'Type',
        align: 'center',
        width: 120,
        render: (value, row, index) => {
            return {children: _.capitalize(row.type)}
        }
    }, {
        title: 'Scale',
        align: 'center',
        width: 60,
        render: (value, row, index) => {
            return {children: _.capitalize(row.scale)}
        }
    }, {
        title: '#Nodes',
        align: 'right',
        width: 80,
        render: (value, row, index) => {
            return {children: numberWithCommas(row.nodes)}
        }
    }, {
        title: '#Edges',
        align: 'right',
        width: 80,
        render: (value, row, index) => {
            return {children: numberWithCommas(row.edges)}
        }
    }, {
        title: '#Features',
        align: 'right',
        width: 80,
        render: (value, row, index) => {
            return {children: numberWithCommas(row.features)}
        }
    }, {
        title: '#Classes',
        align: 'right',
        width: 80,
        render: (value, row, index) => {
            return {children: numberWithCommas(row.classes)}
        }
    }, {
        title: 'Avg. Degree',
        align: 'right',
        width: 100,
        render: (value, row, index) => {
            return {children: row.avg_degree.toFixed(2)}
        }
    }, {
        title: <Tooltip title="Average degree under different difficulties of attacks (Easy/Medium/Hard/Full)">Avg. Degree (E/M/H/F)</Tooltip>,
        align: 'right',
        width: 180,
        render: (value, row, index) => {
            return {children: ['easy', 'medium', 'hard', 'full'].map(d => row.avg_degrees[d].toFixed(2)).join('/')}
        }
    }, {
        title: 'Feature Range',
        align: 'right',
        width: 120,
        render: (value, row, index) => {
            return {children: <span>{row.feature_range[0].toFixed(2)}~{row.feature_range[1].toFixed(2)}</span>}
        }
    }, {
        title: <Tooltip title="normalized by arctan">Feature Range (Norm)</Tooltip>,
        align: 'right',
        width: 150,
        render: (value, row, index) => {
            return {children: <span>{row.feature_range_arctan_norm[0].toFixed(2)}~{row.feature_range_arctan_norm[1].toFixed(2)}</span>}
        }
    }]
}

export const AppDatasets = ({history}) => {
    const [datasets, setDatasets] = useState([])
    const [loading, setLoading] = useState(false)
    useEffect(() => {
        setLoading(true)
        fetch(`${configurations.GITHUB_PROXY_URL}/results/meta/datasets.json`).then(resp => resp.json())
            .then(data => {
                setDatasets(data)
                setLoading(false)
            })
    }, [])
    const columns = getDatasetTableColumns()
    return <div className="app-container app-datasets">
        <Title style={{ textAlign: 'center' }}>Datasets</Title>
        <Divider style={{ marginTop: 50, fontSize: 24 }}>Statistics</Divider>
        <Table loading={loading} columns={columns} dataSource={datasets} bordered pagination={false} scroll={{ x: 1600, y: '80vh' }}/>
        <Divider style={{ marginTop: 50, fontSize: 24 }}>Details</Divider>
        {datasets.map(dataset => {
            return <div className="dataset-card" key={dataset.name}>
                <Title level={4} id={dataset.name}>
                    <i>grb-{dataset.name.toLowerCase()}</i>
                    <Tooltip title="Go to leaderboard">
                        <sup><a onClick={() => history.push(`/leaderboard/${dataset.name.toLowerCase()}`)}><BarChartOutlined /></a></sup>
                    </Tooltip>
                    {dataset.link && <Tooltip title="Download data">
                        <sup><a href={dataset.link}><CloudDownloadOutlined /></a></sup>
                    </Tooltip>}
                </Title>
                {dataset.description && <Title level={5}>Description</Title>}
                {dataset.description && <Paragraph>{dataset.description}</Paragraph>}
                {dataset.refs && <Title level={5}>References</Title>}
                {dataset.refs && dataset.refs.map((ref, idx) => <Paragraph key={idx}>[{idx+1}] {ref.mla}
                    {ref.url && <Tooltip title="View paper">
                        <sup><a href={ref.url}><FilePdfOutlined /></a></sup>
                    </Tooltip>}
                </Paragraph>)}
            </div>
        })}
    </div>
}