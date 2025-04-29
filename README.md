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

**References 📖**

[1] K. Kotowski K, C. Haskamp, J. Andrzejewski, B. Ruszczak, J. Nalepa, D. Lakey, P. Collins, A. Kolmas, M. Bartesaghi, J. Martinez-Heras, G. De Canio, “European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry,” 2024, link: https://arxiv.org/abs/2406.17826.

[2] G. De Canio, J. Eggleston, J. Fauste, A. M. Palowski, and M. Spada, “Development of an actionable AI roadmap for automating mission operations,” in 2023 SpaceOps Conference, Dubai, United Arab Emirates, link: https://opsjournal.org/DocumentLibrary/Uploads/Development%20of%20an%20actionable%20AI%20roadmap%20for%20automating%20mission%20operations.pdf

[3] M. El Amine Sehili and Z. Zhang, “Multivariate Time Series Anomaly Detection: Fancy Algorithms and Flawed Evaluation Methodology,” in Performance Evaluation and Benchmarking, R. Nambiar and M. Poess, Eds., Cham: Springer Nature Switzerland, 2024, pp. 1–17. link: https://link.springer.com/content/pdf/10.1007/978-3-031-68031-1_1.pdf.
