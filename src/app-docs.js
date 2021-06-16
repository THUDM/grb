// deprecated

import React, {  } from 'react'

import configurations from './configurations'
import { MarkdownPage } from './markdown-page'

export const AppDocs = () => {
    return <MarkdownPage url={`${configurations.GITHUB_PROXY_URL}/master/README.md`}/>
}