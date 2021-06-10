![Graph Robustness Benchmark](./public/logo.png)

You can access the website of GRB [here](https://cogdl.ai/grb/).

This website is created with [Create React App](https://facebook.github.io/create-react-app/) + [Ant Design](https://ant.design).

## Develop

If you want to develop the website of GRB, you need to install **npm** (**yarn** optional) first. Then run the following command.

```bash
$ npm install
$ npm start
```

or:

```bash
$ yarn
$ yarn start
```

## Data

Currently, most of the metadata for models, attacks and datasets plus all the experiment results are retrieved from the branches on Github dynamically (though reverse proxied by cogdl since github sometimes gets blocked for users in China mainland). Therefore, if you only need to update the descriptions of models/attacks/datasets or update the experiment results, you only need to modify the data in other branches and do not need to edit the website code.

- [docs](https://github.com/THUDM/grb/tree/docs) for introductions.
- [results](https://github.com/THUDM/grb/tree/results) for experiment results and other metadata.

## Deploy

If you edit the website code and want to deploy the code, you need to run `npm run deploy` or `yarn deploy` first to build the website into `build/` folder. Then you can contact [@cenyk1230](https://github.com/cenyk1230) or [@Somefive](https://github.com/Somefive) for more server deploy details.