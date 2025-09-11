**ESA SPACE ANOMALY DETECTION ALGORITHM**

This repository contains the code and data of algorithms for the detecting of the anomalies in spacecraft telemetry, based on real sensor measurements from a European Space Agency (ESA) mission. The project focuses on identifying anomalous time points in a multivariate time series dataset while minimizing false alarms. [Originated from a **Kaggle** challenge by **ESA**]

**Key Highlights**

**Evaluation Metric:** Uses a corrected event-wise F0.5 score, prioritizing precision over recall. Each anomalous segment is treated as a single event.
**Dataset:**
train.parquet: 14 years of annotated ESA-Mission1 data.
train.csv: 14 years of annotated ESA-Mission1 data, splitted in 3: train test (70%), validation set (15%) and test set (15%) [converted in .csv]
test.parquet: A 6-month unseen fragment for evaluation.
test.csv: A 6-month unseen fragment for evaluation [converted in .csv]
target_channels.csv: 58 monitored channels.
sample_submission.parquet: Format reference for submissions.

**Dataset Description**

The dataset represents several years of real sensor measurements collected from a large spacecraft operated by the European Space Agency. The reduced dataset has been used. It comprises only 6 channels (from 41 to 46), no telecommands and a full lenght of 1.000.000 rows.

The task of the Algorithms is to find all anomalous time points in the test set (binary decision, 0 - normal point, 1 - anomalous point) while minimizing the number of false alarms.

**Events to detect**

ESA-ADB dataset contains three categories of events: anomalies, nominal rare events, and communication gaps. The Algorithm's task is to detect both anomalies and rare nominal events.

**✅ Goal**

Develop models that detect anomalies with high precision and compact detections, avoiding excessive false positives.

**References**
[1] K. Kotowski, C. Haskamp, B. Ruszczak, J. Andrzejewski, and J. Nalepa, “Annotating Large Satellite Telemetry Dataset For ESA International AI Anomaly Detection Benchmark,” in Proceedings of the 2023 conference on Big Data from Space, Vienna: Publications Office of the European Union, Nov. 2023, pp. 341–344. link: https://op.europa.eu/en/publication-detail/-/publication/10ba86b1-7c63-11ee-99ba-01aa75ed71a1/language-en.

**References 📖**

[1] K. Kotowski K, C. Haskamp, J. Andrzejewski, B. Ruszczak, J. Nalepa, D. Lakey, P. Collins, A. Kolmas, M. Bartesaghi, J. Martinez-Heras, G. De Canio, “European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry,” 2024, link: https://arxiv.org/abs/2406.17826.

[2] G. De Canio, J. Eggleston, J. Fauste, A. M. Palowski, and M. Spada, “Development of an actionable AI roadmap for automating mission operations,” in 2023 SpaceOps Conference, Dubai, United Arab Emirates, link: https://opsjournal.org/DocumentLibrary/Uploads/Development%20of%20an%20actionable%20AI%20roadmap%20for%20automating%20mission%20operations.pdf

[3] M. El Amine Sehili and Z. Zhang, “Multivariate Time Series Anomaly Detection: Fancy Algorithms and Flawed Evaluation Methodology,” in Performance Evaluation and Benchmarking, R. Nambiar and M. Poess, Eds., Cham: Springer Nature Switzerland, 2024, pp. 1–17. link: https://link.springer.com/content/pdf/10.1007/978-3-031-68031-1_1.pdf.
