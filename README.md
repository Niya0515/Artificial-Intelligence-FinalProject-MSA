# Artificial-Intelligence-FinalProject-MSA

This is the repository of the final project for Contemporary Artificial Intelligence Course, dealing with the task of multimodal sentiment analysis.



## Setup

This implemetation metation is based on Python3. To run the code, a part of main dependencies are listed below:

- clip==1.0
- ekphrasis==0.5.4
- ftfy==6.1.1
- pandas==1.4.3
- regex==2022.7.9
- scikit_learn==1.1.1
- torch==1.7.0
- torchvision==0.8.1
- transformers==4.20.1

You can simply run:

```bash
pip install -r requirements.txt
```



## Repository structure

```
|-- train 
    |-- sentiment.py # the main code
    |-- utils.py
|-- data_process
    |-- image_process.py
    |-- text_process.py
    |-- utils.py
|-- features 
    |-- clip_ht1.json
    |-- emotion.json
    |-- imagenet.json 
    |-- places.json
    |-- robertabase_ht1.json
|-- pretrained_models
    |-- best_emo_resnet50.pt 
    |-- resnet101_places_best.pth.tar
|-- saved_models
|-- data
    |-- splits
        |-- train.txt
        |-- test.txt
        |-- val.txt
    |-- filenames.txt
    |-- final_test.txt
    |-- image_with_text # You should put data into this directory.
```



## Run pipeline 

1. Download pretrained models to `pretrained_models` directory： [resnet-101](https://drive.google.com/file/d/1ARP8GS5LMGYc8T8lFTuYkBl9I9kJoIiL/view?usp=sharing), [resnet-50](https://drive.google.com/file/d/1sWx3ze8XfZEGf-kPcmiYpY9EOzugdzgu/view?usp=sharing).
2. Put text and image data in the  `data/image_with_text` directory.
2. Run the following scripts to extract features:

- Extract image features: 

```bash
python data_process/image_process.py --vtype clip --ht True
```

- Extract text features: 

```bash
python data_process/text_process.py --btype robertabase --ht True
```

4. Run the following scripts to reproduce the results of the experiment：

* Multimodal sentiment analysis:  

```bash
python train/sentiment.py --vtype clip --ttype clip --ht True --bs 32 --epochs 100
```

* Ablation experiments containing only images：

```bash
python train/sentiment.py --vtype clip --ttype none --ht True --bs 32 --epochs 100
```

* Ablation experiments containing only texts：

```bash
python train/sentiment.py --vtype none --ttype robertabase --ht True --bs 32 --epochs 100
```

Besides, you can specify:

* vtype: the type of visual features to be extracted

  * options: imagenet, places, emotion, clip

* ttype: the type of text features to be extracted

  * options: bertbase, robertabase, clip

* ht: whether to add hash tags during text pre-processing

  * options: True, False

* layer: the layer used to extract text features, default='sumavg'

  * options: sumavg, 2last, last

* bs: batch size, default='32'

* epochs: the number of passes of the entire training dataset, default='100'

* lr: learning rate, default='2e-5'

* norm: whether to add a regularization layer

  * options: 0, 1

  

## References

[1] Tao Jiang, Jiahai Wang, Zhiyue Liu, and Yingbiao Ling. 2020. Fusion-Extraction Network for Multimodal Sentiment Analysis. In Advances in Knowledge Discovery and Data Mining - 24th Pacific-Asia Conference, PAKDD 2020, Singapore, May 11-14, 2020. Springer, 785–797. 

[2] Nan Xu and Wenji Mao. 2017. MultiSentiNet: A Deep Semantic Network for Multimodal Sentiment Analysis. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management, CIKM 2017, Singapore, November 06 10, 2017. ACM, 2399–2402. https://doi.org/10.1145/3132847.313314

[3] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. CoRR abs/2103.00020 (2021). arXiv:2103.00020 https://arxiv.org/abs/2103.00020

[4] Shaojing Fan, Zhiqi Shen, Ming Jiang, Bryan L. Koenig, Juan Xu, Mohan S.Kankanhalli, and Qi Zhao. 2018. Emotional Attention: A Study of Image Sentiment and Visual Attention. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018. IEEE Computer Society, 7521–7531. https://doi.org/10.1109/CVPR.2018.00785

[5] Gullal S. Cheema and Sherzod Hakimov and Eric Muller-Budack and Ralph Ewerth. A Fair and Comprehensive Comparison of Multimodal Tweet Sentiment Analysis Methods. In Proceedings of the 2021 Workshop on Multi-Modal Pre-Training or Multimedia Understanding, Taipei, Taiwan, August 21, 2021. https://doi.org/10.1145/3463945.3469058

[6] Zhen Li, Bing Xu, Conghui Zhu, and Tiejun Zhao. CLMLF:a contrastive learning and multi-layer fusion method for multimodal sentiment detection. In Findings of the Association for Computational Linguistics: NAACL 2022. Association for Computational Linguistics, 2022. arXiv: 2204.05515 https://arxiv.org/abs/2204.05515
