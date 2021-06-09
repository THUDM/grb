import React, {  } from 'react'
import configurations from './configurations'
import { useParams } from 'react-router'
import { MarkdownPage } from './markdown-page'

export const AppIntro = () => {
    const { entry } = useParams()
    return <MarkdownPage url={`${configurations.GITHUB_PROXY_URL}/docs/intro/${entry}.md`}/>
}