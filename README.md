# Exploretransform
> Explore and transform your datasets



[![NPM Version][npm-image]][npm-url]
[![Build Status][travis-image]][travis-url]
[![Downloads Stats][npm-downloads]][npm-url]

Exploretransform is a collection of data exploration functions and custom pipline trasformers.  It's aims to streamline exploratory data analysis and extend some of scikit's data transformers.
&nbsp;

## Installation

Python PYPI:

```sh
!pip install exploretransform
```
&nbsp;
	
## How to use exploretransform

Import the exploretransform package and load the included Boston housing corrected dataset:


```python
import exploretransform as et
```


```python
df, X, y = et.loadboston()
```


&nbsp;


### Summary of Functions and Classes


Function / Class | Description
:---- | :------------- 
nested | takes a list, series or dataframe and returns the location of nested objects
loadboston | loads the Boston housing dataset
glimpse | provides dtype, levels, and first five observations for a dataframe
describe | provides various statistics on a dataframe (zeros, inf, missing, levels, dtypes)
freq | for categorical or ordinal features, provides the count, percent, and cumulative percent for each level
plotfreq | provides a bar plot using the data generated by freq
corrtable | generates a table of all pairwise correlations and uses the average correlation for the row and column in to decide on potential drop/filter candidates
calcdrop | analyzes corrtable output determines which features should be filtered/drop 
skewstats | returns the skewness statistics and magnitude for each numeric feature
ascores | calculates various association scores (kendall, pearson, mic, dcor, spearman) between predictors and target
ColumnSelect | custom transformer that selects columns for pipeline
CategoricalOtherLevel | custom transformer that creates "other" level in categorical / ordinal data based on threshold
CorrelationFilter | custom transformer that filters numeric features based on pairwise correlation

&nbsp;

### Example: describe()


```python
et.describe(X)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>obs</th>
      <th>q_zer</th>
      <th>p_zer</th>
      <th>q_na</th>
      <th>p_na</th>
      <th>q_inf</th>
      <th>p_inf</th>
      <th>dtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>town</td>
      <td>506</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>object</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lon</td>
      <td>506</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lat</td>
      <td>506</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>crim</td>
      <td>506</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>4</th>
      <td>zn</td>
      <td>506</td>
      <td>372</td>
      <td>73.52</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>5</th>
      <td>indus</td>
      <td>506</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>6</th>
      <td>chas</td>
      <td>506</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>category</td>
    </tr>
    <tr>
      <th>7</th>
      <td>nox</td>
      <td>506</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>8</th>
      <td>rm</td>
      <td>506</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>9</th>
      <td>age</td>
      <td>506</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>10</th>
      <td>dis</td>
      <td>506</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>11</th>
      <td>rad</td>
      <td>506</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>category</td>
    </tr>
    <tr>
      <th>12</th>
      <td>tax</td>
      <td>506</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ptratio</td>
      <td>506</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>14</th>
      <td>b</td>
      <td>506</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>15</th>
      <td>lstat</td>
      <td>506</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>float64</td>
    </tr>
  </tbody>
</table>
</div>



&nbsp;

Column | Description
:---- | :------------- 
variable | name of variable
obs | number of observations
q\_zer | number of zeros
p\_zer | percentage of zeros
q\_na | number of missing
p\_na | percentage of missing
q\_inf | number of infinity
p\_inf | percentage of infinity
dtype | Python dtype

&nbsp;

## Development setup

Describe how to install all development dependencies and how to run an automated test-suite of some kind. Potentially do this for multiple platforms.

```sh
make install
npm test
```
&nbsp;

## Release History

* 0.1.0
    * First release

&nbsp;

## Meta

Brian Pietracatella – bpietrac@gmail.com

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/bxp151/exploretransform](https://github.com/bxp151/exploretransform)

&nbsp;

## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki