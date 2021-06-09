import './markdown-page.less'
import { Spin } from 'antd'
import React, { useState, useEffect } from 'react'

import ReactMarkdown from 'react-markdown'
import gfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'

export const MarkdownPage = ({url}) => {
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
    return <div className="app-container app-markdown-page">
      <Spin spinning={loading}>
        <ReactMarkdown remarkPlugins={[gfm]} components={{
            code({node, inline, className, children, ...props}) {
              const match = /language-(\w+)/.exec(className || '')
              const lang = match ? match[1] : 'bash'
              return <SyntaxHighlighter language={lang} PreTag="div" children={String(children).replace(/\n$/, '')} {...props} />
            }
          }}
        >{data}</ReactMarkdown>
      </Spin>
    </div>
}