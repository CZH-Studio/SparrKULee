# SparrKULee数据集下载爬虫

想看到所有的数据，详见[SparrKULee数据集网站](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/K3VSND)

由于该网站在下载多个数据时不会提供文件总大小，因此无法直接使用IDM这类下载器直接下载，但是对于单个文件可以。因此可以采取两步策略：首先获取每一个文件的下载地址，然后对于每一个文件都使用IDM单独下载。

## 1. 获取所有文件的下载地址

首先，需要向`https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/K3VSND`发送get请求，在`<script>`字段即可看到下载链接，导出这些链接到`dataset_link.txt`

## 2. 下载所有文件

请使用IDM导入得到的`dataset_link.txt`，下载