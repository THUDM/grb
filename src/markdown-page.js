import './markdown-page.less'
import { Spin } from 'antd'
import React, { useState, useEffect } from 'react'

import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import gfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import rehypeRaw from 'rehype-raw'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'

export const MarkdownComponent = ({data}) => {
    return <ReactMarkdown
        remarkPlugins={[gfm, remarkMath]}
        rehypePlugins={[rehypeRaw, rehypeKatex]}
        components={{
        code({node, inline, className, children, ...props}) {
        const match = /language-(\w+)/.exec(className || '')
        const lang = match ? match[1] : 'bash'
        return <SyntaxHighlighter language={lang} PreTag="div" children={String(children).replace(/\n$/, '')} {...props} />
        }
    }}>{data}</ReactMarkdown>
}

export const MarkdownLoader = ({url}) => {
    const [data, setData] = useState("")
    const [loading, setLoading] = useState(false)
    useEffect(() => {
        setLoading(true)
        fetch(url)
        .then(resp => resp.text()).then(text => {
            setLoading(false)
            setData(text)
        })
    }, [url])
    return <Spin spinning={loading}>
        <MarkdownComponent data={data}/>
    </Spin>
}

export const MarkdownPage = ({url}) => {
    return <div className="app-container app-markdown-page">
      <MarkdownLoader url={url}/>
    </div>
}