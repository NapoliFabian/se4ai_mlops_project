# MACHINE LEARNING CANVAS

**Designed For:** Fake News Classification
**Designed By:** Napoli Fabian  
**Date:** 19/04/2026  
**Iteration:** 1

**Prediction Task**

The goal of this project is to address a binary classification problem centered on the automatic identification of news veracity. The model is designed to analyze individual news articles —defined as textual objects consisting of both a headline and a body text— to evaluate their factual reliability. Each article represents a single instance where the system processes the combined textual content to determine its trustworthiness.

The classification framework defines two distinct categories:
- *True (Class 0)*: Assigned when the headline and body text are factually accurate and objectively verifiable.
- *False (Class 1)*: Assigned when the content includes incorrect, misleading, or unverified information, regardless of whether the inaccuracy is intentional (disinformation) or accidental (misinformation).

The model is designed for integration into live news feeds or streaming pipelines: in the inference phase, as new articles are added, the system automatically parses the headline and body to produce a label in real time. This allows for the immediate flagging of content, enabling users to distinguish between legitimate information (Class 0) and false claims (Class 1) as they appear in the feed

**Decisions**
The system converts model predictions into reccomandations by providing both a *classification label* and an associated *confidence score*. For each analyzed news article, the model outputs a binary decision (true or false) together with a probability value that quantifies the estimated reliability of the content.
This confidence score enables a more informed interpretation of the prediction:High-confidence classifications indicate that the model is strongly certain about the veracity or falsity of the news, while lower scores suggest uncertainty and encourage additional verification.
This provides end-users with clear recommendations on which news items can be considered reliable and which should be treated with caution or avoided

**Value Proposition**
The primary beneficiaries of the proposed solution are everyday internet users who regularly consume online information through news websites, blogs, and social media platforms. These users are frequently exposed to large volumes of content of varying reliability and often lack the time or expertise required to manually verify the accuracy of each news item.
One of the most critical challenges addressed by this system is misinformation and disinformation. The widespread presence of inaccurate or misleading content makes it difficult for users to distinguish true information from false or manipulated reports. Although manual fact-checking is possible, it is typically time-consuming, cognitively demanding, and impractical at scale. As a result, users may unknowingly rely on unreliable sources or develop mistrust toward online information in general.
The proposed ML-based solution mitigates these issues by providing automatic and immediate credibility assessments, reducing the effort required for verification and supporting faster, more informed decision-making.
In addition to individual users, application developers and digital platforms—such as social media services or online news-applications represent a second category of beneficiaries. These stakeholders can integrate the model directly into their systems via APIs or backend services to automatically monitor user-generated or published content. In this context, the predictions enable platform-level actions such as flagging potentially false information, issuing warnings, limiting content visibility, or blocking and removing posts, depending on the moderation policies defined by the provider.
From an operational perspective, the system is designed to integrate seamlessly into the user’s existing workflow. It can be deployed either as a browser extension, which automatically analyzes articles while the user navigates the web, or as a dedicated web application featuring a simple interface where users can submit or paste news content for evaluation.

**Data Collection**
The initial dataset was constructed by combining public available fake-news detection dataset, with the objective of increasing data diversity and improving the model’s generalization capability. Aggregating heterogeneous sources reduces the risk of overfitting to the linguistic or stylistic patterns of a single dataset and better reflects the variability of real-world online content.
The entities consist of real news articles and social media posts that were originally published online and later collected in public repositories. Labels were inherited from the original datasets and generated through different annotation procedures.
In particular, part of the data originates from the LIAR dataset, whose labels (e.g. false, barely-true, half-true, mostly-true, true) were assigned by professional fact-checkers from PolitiFact. Additional samples come from FakeNewsNet, which aggregates content verified by PolitiFact and GossipCop and provides binary labels (fake/real). Other sources rely on a mix of manual fact-checking, crowdsourcing, or author-provided annotations.
To ensure consistency across datasets, all labels were harmonized into a unified binary schema, where:
- 1 → Fake
- 0 → Real

Multi-class labels from LIAR were mapped into this binary structure, while datasets already providing binary annotations were preserved or converted accordingly. This normalization step simplifies the learning task and enables joint training across all sources.

However, this aggregation introduces several practical considerations: the resulting dataset is not fully homogeneous, as labels originate from different verification criteria and annotation methodologies. Consequently, labeling quality may vary across sources, and the merging process can introduce noise, bias, or inconsistencies that may affect model performance.
To maintain data freshness over time news articles can be periodically collected through automated API pulls or web scraping pipelines, while labeling is performed selectively using a combination of trusted fact-checking services, semi-automated filtering, and limited manual review. This approach balances scalability and reliability, ensuring that the dataset remains up to date without requiring continuous full-scale manual annotation. 

**Data Sources**

The primary dataset used in this project originates from the work presented in the paper “WELFake: Word Embedding over Linguistic Features for Fake News Detection”. The corresponding data collection is public available and was obtained from Kaggle.

**Impact Simulation**

A correct classification clearly represents the optimal outcome, as it allows users or platforms to rely on accurate credibility assessments. However, as with any machine learning model especially if not continuously updated, misclassifications are inevitable and must be evaluated in terms of their impact.
The costs associated with errors are asymmetric.
In particular:
- *False negatives (fake news classified as real)* represent the most critical failure, as they may lead users to trust and disseminate misinformation. This directly amplifies the spread of false or harmful content and undermines the system’s purpose;

- *False positives (real news classified as fake)* are less harmful in comparison, as they may only cause temporary skepticism or additional verification steps;

For this reason, the system is designed to prioritize minimizing false negatives, even at the expense of slightly increasing false positives. From a risk-management perspective, it is better to flag a trustworthy article for further review rather than incorrectly validate misleading information.
To support better decision-making, the model outputs also a confidence score. This value quantifies the certainty of the prediction and allows both users and platforms to interpret results cautiously when confidence is low. Without such uncertainty estimation, the system could replicate the same problem as misinformation itself—providing unreliable guidance.

**Making Predictions**

For individual end-users, predictions are performed in real time. When a user accesses or submits a news article, the system immediately processes the text, extracts features, and returns a credibility classification along with a confidence score. This interaction requires low latency to ensure a smooth user experience. The end-to-end processing time is expected to remain within a few seconds. On the other hand platform-level integrations, such as social media or news aggregation services, predictions needs to be executed in batch mode. In this setting, large volumes of content are periodically analyzed offline to enable large-scale moderation, ranking, or filtering.
Inference will be performed on CPUs, as prediction requires lower computational cost and allows lightweight, scalable, and real-time deployment across standard user devices and servers.

**Building Models**

In production, a single primary classification model is deployed to handle predictions tasks. Maintaining one main model simplifies monitoring, maintenance, and scalability, while ensuring consistent behavior across all user interfaces.
The model should be updated periodically to account for evolving language patterns, emerging misinformation strategies, and newly collected data. Retraining can be scheduled at regular intervals (e.g., weekly or monthly) or triggered when a sufficient amount of new labeled data becomes available. This continuous update process helps prevent performance degradation and concept drift.
For model training, GPU resources will be used to accelerate computation and enable faster training and frequent updates as new data becomes available, ensuring that the model can be efficiently retrained and redeployed.

**Features**

The main features are extracted from the text content of the news, including the title, description, and relevant keywords.
Text data are preprocessed, tokenized, and vectorized for model input. Future improvements may include data augmentation and new feature types to enhance performances.

**Monitoring** 

System performance is tracked using standard metrics such as Accuracy, Precision, Recall, and F1-score.
Monitoring is continuous and updated after each retraining or dataset refresh to detect performance drift.
Operational indicators, including triage time reduction and correct assignment rate, are also reviewed to ensure long-term model stability and effectiveness.
