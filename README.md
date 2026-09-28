# CPO产业链财报 RAG 问答知识库

## 项目简介
基于 10 家 CPO 产业链上市公司 2026 年半年度报告，构建向量 + BM25 混合检索的问答系统，答案带出处。

## 公司列表
中际旭创、新易盛、剑桥科技、华工科技、天孚通信、光迅科技、仕佳光子、长飞光纤、亨通光电、烽火通信。

## 运行步骤
1. 安装依赖：`pip install -r requirements.txt`
2. 将 10 份 PDF 放入 `pdf_reports/`
3. 构建索引：`python build_index.py`
4. 设置 DeepSeek API Key：`set DEEPSEEK_API_KEY=你的key`
5. 启动页面：`streamlit run app.py`

## 数据来源
上交所 sse.com.cn、深交所 szse.cn、巨潮资讯 cninfo.com.cn。

## 结论
单公司基础指标题表现好；跨公司全景题 top-k 覆盖不足，需要分组召回或 GraphRAG。