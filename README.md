# Deep Learning Course Assignments & Final Project

This repository contains the assignments and final project for the **Neural Networks and Deep Learning** course.

This course was held in Spring 2025 under the supervision of **Dr. Behnam Bahrak** at [Teias Institute](https://teias.institute/computer-science-department/).

All implementations were developed collaboratively with [Peyman Naseri](https://github.com/peyman886).

---

## Table of Contents

1. [Assignment 1: MLPs, Adaline/Madaline, and Regression Tasks](#assignment-1-mlps-adalinemadaline-and-regression-tasks)  
2. [Assignment 2: CNNs for Skin Lesion Classification + Fine-Tuning](#assignment-2-cnns-for-skin-lesion-classification--fine-tuning)  
3. [Assignment 3: Time Series Forecasting and Suicide Ideation Detection](#assignment-3-time-series-forecasting-and-suicide-ideation-detection)  
4. [Assignment 4: Extractive QA with BERT and Sentiment Analysis with GPT-2](#assignment-4-extractive-qa-with-bert-and-sentiment-analysis-with-gpt-2)  
5. [Assignment 5: Autoencoders, Clustering, and GANs](#assignment-5-autoencoders-clustering-and-gans)  
6. [Assignment 6: Reinforcement Learning with DQN Variants](#assignment-6-reinforcement-learning-with-dqn-variants)  
7. [Final Project: Instruction Fine-Tuning and Image Captioning](#final-project-instruction-fine-tuning-and-image-captioning)  

---

## Homework Assignments

### Assignment 1: MLPs, Adaline/Madaline, and Regression Tasks
- Implement and compare **MLPs on Fashion-MNIST** with different regularization/dropout settings.  
- Train **Adaline and Madaline** classical models on Wine and synthetic datasets.  
- Study **optimization algorithms** (SGD, Adam, Nadam, RMSprop) and hyperparameter tuning.  
- Analyze with **confusion matrices, heatmaps, training curves**.

---

### Assignment 2: CNNs for Skin Lesion Classification + Fine-Tuning
- Use **HAM10000 dataset** (binary subset).  
- Apply **data preprocessing and augmentation**.  
- Train a **CNN from scratch** for lesion classification.  
- Fine-tune **VGG16, ResNet50** on Cats vs Dogs dataset.  
- Evaluate with **accuracy, ROC, precision, recall, F1**.

---

### Assignment 3: Time Series Forecasting and Suicide Ideation Detection
- Predict **crude oil prices** with **RNN, LSTM, BiLSTM, GRU**.  
- Compare with **ARIMA and SARIMA** baselines.  
- Metrics: **RMSE, MAE, MAPE, R²**.  
- NLP task: classify **suicidal ideation in tweets** using preprocessing, embeddings, and **LSTM/CNN+LSTM**.  

---

### Assignment 4: Extractive QA with BERT and Sentiment Analysis with GPT-2
- Build **extractive QA system** (ParsBERT & ALBERT-fa) on **PQuAD dataset**.  
- Evaluate with **Exact Match (EM)** and **F1 score**.  
- Fine-tune **GPT-2** on IMDb dataset for **sentiment classification**.  
- Compare with traditional methods; evaluate with **accuracy, perplexity**.

---

### Assignment 5: Autoencoders, Clustering, and GANs
- Implement **Convolutional Autoencoders** on **MNIST and Fashion-MNIST**.  
- Use encoders as **feature extractors for clustering (KMeans)**.  
- Visualize clustering vs ground truth with **Silhouette score**.  
- Implement **DCGAN** for digit generation (Class-5).  
- Study **GAN stabilization techniques** (Label smoothing, Noise).  

---

### Assignment 6: Reinforcement Learning with DQN Variants
- Task 1: **Inventory management with perishable goods** using **DQN + Reward Shaping**.  
- Implement custom Gym environments and compare **Base-stock, BSP-low-EW shaping** vs vanilla DQN.  
- Task 2: **Robot path planning** using **Dueling DQN, M-DQN, DM-DQN**.  
- Evaluate with **learning curves, optimality gap, steady-state distributions**.  

---

## Final Project: Instruction Fine-Tuning and Image Captioning

### Part 1: Instruction Fine-Tuning (IFT) of LLMs
- **Dataset**: SlimOrca (Persian ~50k conversations, GPT-4 refined).  
- **Models**: Gemma-2B, Llama-3.2-3B (Base vs Instruct).  
- **Methods**:  
  - **Soft Prompts** (Prompt Tuning, Prefix Tuning, P-Tuning).  
  - **LoRA** and variants (DoRA, LoHa, RsLoRA).  
  - **Partial Layer Fine-Tuning** (freezing all but last layers).  
- **Tasks**:  
  - Data preparation and tokenizer setup.  
  - Implement fine-tuning with PEFT.  
  - Compare **accuracy, F1, memory usage, training time** across methods.  
  - Analyze **advantages, drawbacks, and resource efficiency**.  

### Part 2: Image Captioning
- **Dataset**: Flickr8k (images + captions).  
- **Models**:  
  - **CNN-RNN (EfficientNet-B0 + LSTM/GRU)**.  
  - Add **Attention mechanism** for improved focus.  
  - Alternative: **CNN-Transformer** decoder.  
- **Evaluation**:  
  - **BLEU-1 to BLEU-4** scores.  
  - Qualitative caption analysis with generated examples.  
  - Visualization of **attention heatmaps** for interpretability. 
