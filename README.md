Your objective is to design an algorithm to detect anomalies in a multivariate time series setting. The data used in the Challenge is real spacecraft data coming from one of the European Space Agency's missions (https://www.esa.int/).

You have access to a large set of historical data, baseline algorithms, and detailed descriptions.

The Spacecraft Anomaly Challenge on ESA dataset starts now! 🚀

**Evaluation metric ✅**

Submissions are evaluated using the corrected event-wise F0.5 score for anomaly detection in time series introduced by Sehili & Zhang [3] and adopted as a primary metric in the ESA-ADB benchmark [1]. The metric is a harmonic mean of corrected event-wise precision and event-wise recall, with precision having 2 times higher importance than recall:

F0.5=(1+0.52)∗Precisionecorr∗Recalle0.52∗Precisionecorr+Recalle

Event-wise precision and recall treat each continuous anomalous segment as a single event, regardless of its size. A segment is considered an event-wise true positive TPe if at least one of its samples is detected; otherwise, it is an event-wise false negative FNe. Any continuous detection that does not overlap with an anomalous segment is counted as an event-wise false positive FPe. Additionally, due to the correction applied to the event-wise precision, each falsely detected sample FPt decreases the value of the corrected event-wise precision relatively to the number of nominal sample Nt.

Precisionecorr=TPeTPe+FPe∗(1−FPtNt)

Recalle=TPeTPe+FNe

In summary, the proposed solutions should focus on:

minimizing the number of event-wise and sample-wise false positives
producing short and compact detections (not necessarily covering the whole anomalous segment, it is enough to detect at least one sample)
The metric does NOT care about:

the level of overlap between correct detections and anomalous segments
detection timing

**Dataset Description**

**TLDR**

The dataset represents several years of real sensor measurements collected from a large spacecraft operated by the European Space Agency. It is a single long continuous multivariate time series data of 87 mission-critical parameters. The dataset was carefully annotated for anomalies and rare events based on the original flight control reports and several iterations of manual and algorithm-driven refinement of labels [1].

Your task is to find all anomalous time points in the test set (binary decision, 0 - normal point, 1 - anomalous point, see the sample_submission.parquet file) while minimizing the number of false alarms.

**List of files**

train.parquet - the training set; 14 years of ESA-Mission1 data from the ESA-AD dataset with annotated anomalous time points (is_anomaly column)
test.parquet - the test set; a never before published 6-months long fragment of ESA-Mission1 data
target_channels.csv - a list of names of 58 channels monitored for anomalies (other 29 channels are just external auxiliary variables)
sample_submission.parquet - a sample submission file in the correct format

**Preprocessing**

The dataset is a simplified and preprocessed version of the raw ESA-Mission1 data from the ESA-AD dataset publicly available on Zenodo. The preprocessing is described together with all other dataset details in the ESA-ADB paper [2] and the preprocessing script is available on the official ESA-ADB GitHub repository here. For simplicity, timestamps have been replaced with integer index and annotations have been aggregated across all channels into the single is_anomaly column, such that 1 is marked for a time point if at least one channel is affected by an anomaly or a rare event. Participants are welcome to propose their own preprocessing and annotations aggregation methods, however, the test set is already preprocessed and aggregated using our described methods.

**Channels**

The dataset contains 87 channels, among which 58 are target channels (monitored for anomalies) listed in the target_channels.csv file, 18 channels are auxiliary environmental variables (not monitored for anomalies), and 11 are telecommands (binary control commands sent by operators; marked with the telecommand_ prefix). The subset of 6 channels 41-46 is recommended as a good starting point for developing and testing algorithms before applying them to the full set.

These 87 channels are just a small subset of real spacecraft telemetry with thousands of channels. However, this subset was designed by spacecraft operators to be a self-contained representation of the most important mission-critical and auxiliary variables necessary to identify anomalies.

**Events to detect**

ESA-ADB dataset contains three categories of events: anomalies, nominal rare events, and communication gaps. The test set does not contain communication gaps. Your task is to detect both anomalies and rare nominal events.

**References**
[1] K. Kotowski, C. Haskamp, B. Ruszczak, J. Andrzejewski, and J. Nalepa, “Annotating Large Satellite Telemetry Dataset For ESA International AI Anomaly Detection Benchmark,” in Proceedings of the 2023 conference on Big Data from Space, Vienna: Publications Office of the European Union, Nov. 2023, pp. 341–344. link: https://op.europa.eu/en/publication-detail/-/publication/10ba86b1-7c63-11ee-99ba-01aa75ed71a1/language-en.

**References 📖**

[1] K. Kotowski K, C. Haskamp, J. Andrzejewski, B. Ruszczak, J. Nalepa, D. Lakey, P. Collins, A. Kolmas, M. Bartesaghi, J. Martinez-Heras, G. De Canio, “European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry,” 2024, link: https://arxiv.org/abs/2406.17826.

[2] G. De Canio, J. Eggleston, J. Fauste, A. M. Palowski, and M. Spada, “Development of an actionable AI roadmap for automating mission operations,” in 2023 SpaceOps Conference, Dubai, United Arab Emirates, link: https://opsjournal.org/DocumentLibrary/Uploads/Development%20of%20an%20actionable%20AI%20roadmap%20for%20automating%20mission%20operations.pdf

[3] M. El Amine Sehili and Z. Zhang, “Multivariate Time Series Anomaly Detection: Fancy Algorithms and Flawed Evaluation Methodology,” in Performance Evaluation and Benchmarking, R. Nambiar and M. Poess, Eds., Cham: Springer Nature Switzerland, 2024, pp. 1–17. link: https://link.springer.com/content/pdf/10.1007/978-3-031-68031-1_1.pdf.
