示例中用到了spacy工具对语料分词等，因此需要安装spacy及对应语言语料。
pip install spacy（2.2.0以上，用如下方式安装集加载）
python -m spacy download en
如果网速慢，使用
pip install en_core_web_sm-2.2.0.tar.gz
其他语言同上
载入使用如下方式：
spacy.load('en_core_web_sm'）

