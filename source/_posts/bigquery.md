---
title: 初探 BigQuery
categories:
- 科研
tags:
- 云服务
- Google Cloud
- 数据仓库
- BigQuery
index_img: /images/backgrounds/bigquery.webp
banner_img: /images/backgrounds/bigquery.webp
date: 2023-02-05 20:00:00
---

在近期的科研任务中，我需要对一份亿级的数据集进行分析和处理。

<!-- more -->

在之前的任务中，仅需要逐行读入数据集，输入算法进行分析即可，故当时使用 Python 脚本直接对数据集进行读取，并编写算法进行分析处理。但新的任务涉及使用多个数据集交叉对比分析，需要使用整个数据集的特征，由于数据集大小十分庞大，读取数据集进内存的方法显然已不现实。近期实验室正在探索 BigQuery，故本次试用 BigQuery 进行分析。

# BigQuery 是什么

BigQuery 是谷歌云推出的无服务器企业数据仓库，其能在秒级时间内分析 TB 级的数据，可用于进行大数据分析与处理。

在用户体验上，BigQuery 的基本功能相当于一个规模巨大且速度飞快的 SQL 数据库，只需要导入数据，运行 SQL 语句，即可得到运算结果，并可将其导出到外部存储继续下一步分析。

![BigQuery 由存储集群与计算集群通过高速网络相连](architecture.webp "BigQuery 架构")

在其背后，分布式高可用存储集群与计算集群通过超高速网络相连，每次请求都会被拆解，并由多个 Worker 执行，以获得高效的查询效率。

# 导入数据

BigQuery 支持流式数据导入与批量数据导入。分析使用的数据集是固定的，所以使用批量数据导入的方法进行导入。

BigQuery 批量数据导入支持多种格式，方便本地构造的文件格式有 CSV, 换行符分隔的 JSON 等。其也支持多种数据来源，如本地上传与 Google Cloud Storage。格式方面，CSV 最为方便，但无法支持某些特殊数据类型（如数组），这种情况可以使用 JSON。本地上传小型数据集可以用于测试，但超过 10MB 的数据集只能通过 Google Cloud Storage 进行上传。

上传数据需要连接谷歌服务器。

## gzip 压缩

BigQuery 对上传的文件支持 gzip 压缩。在本地执行以下指令即可对数据进行压缩，上传压缩后的数据可以有效节省上传时间与存储开销。

```bash
# 压缩指令将源文件替换成压缩文件。-v表示输出更多信息，-9表示最小压缩率。
gzip -v9 dataset.csv
# 解压指令
gunzip dataset.csv.gz
```

## 分区与聚簇

分区与聚簇对数据表的存储方式提供了更多调整空间，是有效提高查询效率，降低查询成本的方法。

简单来说，分区是分块，聚簇是排序。

分区是指将一个表按照数据的某个属性分成不同的子表，分别存储。当查询语句需要对某个属性进行筛选时，使用此功能可以让 BigQuery 仅访问该属性对应的存储空间，减少查询开销，提高查询效率。比较常见的应用是在分析日志时按照日志产生的日期进行分区，在查询某天的数据时，BigQuery 仅访问对应的分区，查询字节数将会仅限于当天的日志量。

聚簇是指在表中按照数据的某个或某几个属性排序的顺序进行存储。当查询语句对某属性进行筛选时，使用此功能也可以让 BigQuery 仅访问该属性对应的存储空间，减少查询开销，提高查询效率。

![聚簇表按照某一列进行排序](clustering-tables.webp "聚簇表")

分区和聚簇看起来比较相似，但有各自的优缺点：分区几乎没有负面作用，但分区的使用范围较窄：分区参考的属性只能是下面两种：一种是整数，创建分区表时需事先指定上下界与步长，最多可创建10000个分区子表；另一种是时间戳，可按特定时长进行拆分。聚簇可以在任意列上应用，但聚簇在查询过滤时必须包含第一个聚簇关键字才能生效，否则聚簇表优化将会失效，仍将扫描全表。

![查询不包含第一关键字时聚簇表优化将会失效](optimize-query-clustering-tables.webp "优化聚簇表查询")

分区与聚簇可以结合使用，进一步提升查询速度。

![聚簇表结合分区表可以进一步提升查询速度](clustering-and-partitioning-tables.webp "聚簇表结合分区表")

# 分析数据

导入数据后，通过 Google Standard SQL 语言对数据库进行查询，分析与处理。

Google Standard SQL 提供了一些特殊的数据结构，如数组类型，地理位置类型等，在[官方的参考文档](https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types?hl=zh-cn#array_type)中详细记录了使用方法。

通过 [bq](https://cloud.google.com/bigquery/docs/reference/bq-cli-reference?hl=zh-cn) CLI 工具，可以定期或批量发送查询。但运行时需要注意限额，谷歌云对并行任务与任务频率都有着限制，需要在编码时注意。

如果不慎关闭了查询运行页面，重新进入管理台，点击右侧下方的“个人纪录”或“项目历史记录”，点击对应作业ID-作为新查询打开，即可看到执行状态或执行结果。

运算结果默认保存在临时表里，仅能在查询结果页面看到，可以在查询设置内将结果输出位置更改为指定表中，也可以在查询结果页面点击“保存查询结果”保存到 BigQuery 表或到导出下载到本地。

# 总结

借助 BigQuery ~(与 ChatGPT)~ 的力量，我用更少的代码量与更短的时间便完成了对这一大型数据集的分析与处理。

BigQuery 特别适用于在大数据集上进行简单的分析与处理（使用 SQL 语句即可完成），而对于大规模数据集上的复杂算法，虽然也可以通过在算法中调用 API 进行分析，但限额问题使得在为其编写代码时需要特别考量。对于需要高并发高吞吐的算法，建议使用其他数据仓库进行替代。

![ChatGPT 通过分析实际问题编写了一行正确的 SQL 语句](chatgpt.webp "题外话：ChatGPT yyds（感谢qlgg）")