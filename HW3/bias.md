# Bias and Debiasing in NLP

Papers and Resources for Biases and Debiasing in NLP

by Binyao Wang (bingyao.wang@yale.edu)

## Table of Contents

- [Resources](#resources)
  - [Posts](#posts)
  - [Tasks and Competitions](#tasks-and-competitions)
  - [Other Bibliographies](#other-bibliographies)
- [Papers](#papers)
  - [Detection of Biases](#detection-of-biases)
  - [Measurement of Biases](#measurement-of-biases)
  - [Debiasing Methods](#debiasing-methods)

## Resources

### Posts 
In this section, we list great posts (some of them features awesome 
visualizations) that investigated biases in word embeddings. 
* Bias in Word Embeddings: What Causes It? [[link](https://kawine.github.io/blog/nlp/2019/09/23/bias.html)]
* Racial Bias in BERT (with good visualization) [[link](https://towardsdatascience.com/racial-bias-in-bert-c1c77da6b25a)]
* Gender bias in word embeddings? [[link](https://www.kaggle.com/rtatman/gender-bias-in-word-embeddings)]
* On gender bias in word embeddings [[link](https://medium.com/linguaphile/on-gender-bias-in-word-embeddings-e53c40ba9294)]
* How Word-Embeddings evolved to learn social biases and how to improve them to forget it. [[link](https://medium.com/analytics-vidhya/how-word-embeddings-evolved-to-learn-social-biases-and-how-to-improve-it-to-forget-them-f37e3244d3a5)]
* Need for Fair Word Embeddings in Natural Language Processing [[link](https://medium.com/@pradeeprajkvr/need-for-fair-word-embeddings-in-natural-language-processing-84e52fb8b493)]

### Tasks and Competitions
In this section, we list tasks and competitions that aimed to address
biases in NLP. 
* Jigsaw Unintended Bias in Toxicity Classification [[link](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)]
* Korean Bias Detection [[link](https://www.kaggle.com/c/korean-bias-detection)]
* Korean Gender Bias Detection [[link](https://www.kaggle.com/c/korean-gender-bias-detection)]
* Data Solution Hackathon: Ethical AI [[link](https://www.kaggle.com/c/ds-hackathon-ethical-ai/overview)]

### Other Bibliographies
In this section, we list other bibliographires including survey type 
of papers, as well as bias-relevant categories in Papers with Code. 
* A Survey on Bias in Deep NLP [[link](https://www.mdpi.com/2076-3417/11/7/3184/htm)]
* A Critical Survey of "Bias" in NLP [[link](http://users.umiacs.umd.edu/~hal/docs/daume20power.pdf)]
* Societal Biases in Language Generation: Progress and Challenges [[link](https://arxiv.org/pdf/2105.04054.pdf)]
* Papers with Code: Bias Detection [[link](https://paperswithcode.com/task/bias-detection)]
* Papers with Code: Gender Bias Detection [[link](https://paperswithcode.com/task/gender-bias-detection)]

## Papers

### Detection of Biases
In this section, we list papers that observed and studied biases in either static word embeddings or contextualized embeddings.

* Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings (2016) [[paper](https://arxiv.org/pdf/1607.06520.pdf)]
* Semantics derived automatically from language corpora contain human-like biases (2017) [[paper](https://arxiv.org/pdf/1608.07187.pdf)]
* Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods (*NAACL* 2018) [[paper](https://aclanthology.org/N18-2003.pdf)]
* Adversarial Removal of Demographic Attributes from Text Data (*ACL* 2018) [[paper](https://aclanthology.org/D18-1002.pdf)] [[Github](https://github.com/yanaiela/demog-text-removal)]
* Gender Bias in Coreference Resolution (*NAACL* 2018) [[paper](https://aclanthology.org/N18-2002.pdf)] [[Github](https://github.com/rudinger/winogender-schemas)]
* Learning Gender-Neutral Word Embeddings (*EMNLP* 2018) [[paper](https://arxiv.org/pdf/1809.01496.pdf)] [[Github](https://github.com/uclanlp/gn_glove)]
* Gender Bias in Contextualized Word Embeddings (*NAACL* 2019) [[paper](https://aclanthology.org/N19-1064.pdf)]
* Evaluating the Underlying Gender Bias in Contextualized Word Embeddings (*ACL* 2019) [[paper](https://aclanthology.org/W19-3805.pdf)]
* Entity-Centric Contextual Affective Analysis (*ACL* 2019) [[paper](https://arxiv.org/pdf/1906.01762.pdf)]
* Gender-preserving Debiasing for Pre-trained Word Embeddings (*ACL* 2019) [[paper](https://aclanthology.org/P19-1160.pdf)] [[Github](https://github.com/kanekomasahiro/gp_debias)]
* Mitigating Gender Bias in Natural Language Processing: Literature Review (*ACL* 2019) [[paper](https://aclanthology.org/P19-1159.pdf)]

### Measurement of Biases 
In this section, we list papers that proposed ways/metrics to measure biases. 

* Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings (2016) [[paper](https://arxiv.org/pdf/1607.06520.pdf)]
* Semantics derived automatically from language corpora contain human-like biases (2017) [[paper](https://arxiv.org/pdf/1608.07187.pdf)]
* Word Embeddings Quantify 100 Years of Gender and Ethnic
Stereotypes (2017) [[paper](https://arxiv.org/pdf/1711.08412.pdf)] [[Github](https://github.com/nikhgarg/EmbeddingDynamicStereotypes)]
* Black is to Criminal as Caucasian is to Police: Detecting and Removing Multiclass Bias in Word Embeddings (*NAACL* 2019) [[paper](https://aclanthology.org/N19-1062.pdf)]
* On Measuring Social Biases in Sentence Encoders (*NAACL* 2019) [[paper](https://aclanthology.org/N19-1063.pdf)] [[Github](https://github.com/W4ngatang/sent-bias)]
* Measuring Bias in Contextualized Word Representations (*NAACL* 2019) [[paper](https://aclanthology.org/W19-3823.pdf)] 
* Mitigating Language-Dependent Ethnic Bias in BERT (*EMNLP* 2021) [[paper](https://arxiv.org/pdf/2109.05704.pdf)] [[Github](https://github.com/jaimeenahn/ethnic_bias)]



### Debiasing Methods 
In this section, we list papers that proposed
methods to mitigate or remove biases in language representations(word embeddings)/language models/NLU, NLG tasks. Specfically, a number of papers focused on reducing gender bias, so I categorize this section into mitigiating gender biases and other types of biases. 

Gender:
* Counterfactual Data Augmentation for Mitigating Gender Stereotypes in Languages with Rich Morphology (*ACL* 2019) [[paper](https://aclanthology.org/P19-1161v2.pdf)] 
* Automatic Gender Identification and Reinflection in Arabic (*ACL* 2019) [[paper](https://aclanthology.org/W19-3822v2.pdf)]
* Evaluating Gender Bias in Machine Translation (*ACL* 2019) [[paper](https://aclanthology.org/P19-1164.pdf)] [[Github](https://github.com/gabrielStanovsky/mt_gender)]
* Assessing Gender Bias in Machine Translation -- A Case Study with Google Translate (*Neural Computing and Applications* 2019) [[paper](https://arxiv.org/pdf/1809.02208.pdf)] [[Github](https://github.com/marceloprates/Gender-Bias)]
* Reducing Gender Bias in Word-Level Language Models with a Gender-Equalizing Loss Function (*ACL* 2019) [[paper](https://aclanthology.org/P19-2031.pdf)] 
* Identifying and Reducing Gender Bias in Word-Level Language Models (*NAACL* 2019) [[paper](https://aclanthology.org/N19-3002.pdf)] [[Github](https://github.com/BordiaS/language-model-bias)]
* Gender-Aware Reinflection using Linguistically Enhanced Neural Models (*ACL* 2020) [[paper](https://aclanthology.org/2020.gebnlp-1.12.pdf)] [[Github](https://github.com/CAMeL-Lab/gender-reinflection)]
* Neural Machine Translation Doesn’t Translate Gender Coreference Right Unless You Make It (*ACL* 2020) [[paper](https://aclanthology.org/2020.gebnlp-1.4.pdf)] [[Github](https://github.com/DCSaunders/tagged-gender-coref)]
* Reducing Gender Bias in Neural Machine Translation as a Domain Adaptation Problem (*ACL* 2020) [[paper](https://aclanthology.org/2020.acl-main.690v2.pdf)] [[Github](https://github.com/DCSaunders/gender-debias)]
* Conversational Assistants and Gender Stereotypes: Public Perceptions and Desiderata for Voice Personas (*COLING* 2020) [[paper](https://aclanthology.org/2020.gebnlp-1.7.pdf)] 
* Does Gender Matter? Towards Fairness in Dialogue Systems (*ICCL* 2020) [[paper](https://aclanthology.org/2020.coling-main.390.pdf)] [[Github](https://github.com/zgahhblhc/DialogueFairness)]
* Investigating Gender Bias in Language Models Using Causal Mediation Analysis (*NeurIPS* 2020) [[paper](https://proceedings.neurips.cc/paper/2020/file/92650b2e92217715fe312e6fa7b90d82-Paper.pdf)] [[Github](https://github.com/sebastianGehrmann/CausalMediationAnalysis)]
* They, Them, Theirs: Rewriting with Gender-Neutral English (2021) [[paper](https://arxiv.org/pdf/2102.06788.pdf)] 
* Investigating Failures of Automatic Translation in the Case of Unambiguous Gender (2021) [[paper](https://arxiv.org/pdf/2104.07838.pdf)]
* Gender Bias in Machine Translation (*TACL* 2021) [[paper](https://arxiv.org/pdf/2104.06001.pdf)] 
* Towards Cross-Lingual Generalization of Translation Gender Bias (*FACCT* 2021) [[paper](https://dl.acm.org/doi/pdf/10.1145/3442188.3445907)] [[Github](https://github.com/nolongerprejudice/tgbi-x)]
* Revealing Persona Biases in Dialogue Systems (2021) [[paper](https://arxiv.org/pdf/2104.08728.pdf)] [[Github](https://github.com/ewsheng/persona-biases)]
* Bias Out-of-the-Box: An Empirical Analysis of Intersectional Occupational Biases in Popular Generative Language Models (*NeurIPS* 2021) [[paper](https://arxiv.org/pdf/2102.04130.pdf)] [[Github](https://github.com/oxai/intersectional_gpt2)]
* Investigating Failures of Automatic Translation in the Case of Unambiguous Gender (2021) [[paper](https://arxiv.org/pdf/2104.07838.pdf)] 
* Self-Diagnosis and Self-Debiasing: A Proposal for Reducing Corpus-Based Bias in NLP (*TACL* 2021) [[paper](https://arxiv.org/pdf/2103.00453.pdf)] [[Github](https://github.com/timoschick/self-debiasing)]
* On Mitigating Social Biases in Language Modelling and Generation (*ACL* 2021) [[paper](https://aclanthology.org/2021.findings-acl.397.pdf)]

Others (Some papers address other types of biases in addition to gender biases):
* Mitigating Gender Bias in Natural Language Processing: Literature Review (*ACL* 2019) [[paper](https://aclanthology.org/P19-1159.pdf)]
* The Woman Worked as a Babysitter: On Biases in Language Generation (*EMNLP* 2019) [[paper](https://arxiv.org/pdf/1909.01326.pdf)] [[Github](https://github.com/ewsheng/nlg-bias)]
* PowerTransformer: Unsupervised Controllable Revision for Biased Language Correction (*EMNLP* 2020) [[paper](https://aclanthology.org/2020.emnlp-main.602.pdf)] 
* Reducing Sentiment Bias in Language Models via Counterfactual Evaluation (*EMNLP* 2020) [[paper](https://aclanthology.org/2020.findings-emnlp.7.pdf)] 
* Investigating African-American Vernacular English in Transformer-Based Text Generation (*EMNLP* 2020) [[paper](https://aclanthology.org/2020.emnlp-main.473.pdf)] 
* Persistent Anti-Muslim Bias in Large Language Models (*AIES* 2021) [[paper](https://dl.acm.org/doi/pdf/10.1145/3461702.3462624)] 
* Bias Out-of-the-Box: An Empirical Analysis of Intersectional Occupational Biases in Popular Generative Language Models (*NeurIPS* 2021) [[paper](https://arxiv.org/pdf/2102.04130.pdf)] [[Github](https://github.com/oxai/intersectional_gpt2)]
* Revealing Persona Biases in Dialogue Systems (2021) [[paper](https://arxiv.org/pdf/2104.08728.pdf)] [[Github](https://github.com/ewsheng/persona-biases)]
* Self-Diagnosis and Self-Debiasing: A Proposal for Reducing Corpus-Based Bias in NLP (*TACL* 2021) [[paper](https://arxiv.org/pdf/2103.00453.pdf)] [[Github](https://github.com/timoschick/self-debiasing)]
* Challenges in Automated Debiasing for Toxic Language Detection (*EACL* 2021) [[paper](https://aclanthology.org/2021.eacl-main.274.pdf)] [[Github](https://github.com/XuhuiZhou/Toxic_Debias)]


