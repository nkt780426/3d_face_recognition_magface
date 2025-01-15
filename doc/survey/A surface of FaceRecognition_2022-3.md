# 5. Backbone size and data distribution
Ở phần 4, ta đã có 1 khảo sát toàn diện về các thuật toán FR, nhưng hiếm khi đề cập đến effect của backbone size và sự phân tán của training set trong quá trình train. 
Khác với ở phần 4, các thuật toán được xây dựng cho 1 giả thuyết cụ thể nào đó còn backbone size và sự phân phối của dữ liệu ảnh hưởng trực tiếp đến tất cả các tình huống và hiệu suất của mô hình 
Phần này ta thảo luộn về 3 vấn đề
- backbone size
- data depth and breadth: độ sâu và chiều rộng của dữ liệu
- Long tail distribution:

## 5.1. Backbone size
Ta đã biết rằng, khi train model thì càng nhiều data để train thì thuật toán của model càng được cải thiện. Tuy nhiên với backbone cụ thể, khi training data đạt đến 1 kích thước cụ thể nào đó, hiệu suất của nó sẽ không còn được cải thiện đáng kể bằng cách thêm dữ liệu nữa mà còn tăng thêm chi phí train.

Khảo sát sau hương đến việc tìm ra tác động của việc tăng dữ liệu đến hiệu suất của mô hình. 
    Họ chọn Iresnet50, Iresnet100 và Mobilefacenet làm backbone và lần lượt chọn 10%, 40%, 70% và 100% ids từ dataset Webface42m làm training data.
    Sử dụng Arcface loss và PartialFC operator để đạt được sự hội tụ của model (loss function).
    Sử dụng SGD làm optimizer
    Weight decay (suy giảm trọng số) được đặt thành 5e-4.
    Testset là 4 tập: the LFW, the AgeDB, the CFP-FP, and the IJB-C

Kết quả hình: ![](images/5.1.%20Result.png)
    Với mobilefacenet, khi tỷ lệ training set tăng từ 10-40%, hiệu suất mô hình rõ ràng được cải thiện trên bốn tập dữ liệu thử nghiệm, từ 99,75% lên 99,80% trên LFW, từ 97,13% lên 97,92% trên AgeDB, từ 98,73% lên 98,99% trên CFP-FP và từ 95,29% lên 96,46% trên IJB-C
    Khi tỷ lệ mẫu trên 40%, hiệu suất của Mobilefacenet vẫn ổn định

    Đối với Iresnet50, điểm ngoặt là tỷ lệ mẫu 70%

    Trong khi đó, hiệu suất của Iresnet100 cải thiện đôi chút và liên tục khi dữ liệu đào tạo tăng. Đối với ba xương sống khác nhau, rõ ràng là hiệu suất mô hình được cải thiện khi lượng dữ liệu đào tạo tăng.

## 5.2. Data depth and breadth

Trong quá trình thu thập dữ liệu, cho dù chúng ta chỉ thu thập 1 vài IDs. Nhưng với mỗi IDs, ta thực hiện thu thập rất nhiều ảnh. Đây chính là data depth (độ sâu của dữ liệu). Rõ ràng data depth giúp chúng ta đảm bảo intra-class variations trong 1 IDs. 

Ngược lại, nếu ta thu thập hình ảnh của nhiều IDs và thu thập ít ảnh với mỗi người. Đây chính là breath (độ rông của dữ liệu). Việc này giúp thuật toán được đào tạo bởi đủ các danh tính khác nhau.

Thực tế trong industry, họ dễ dàng cải thiện chiều rộng hơn là chiều sâu của dataset. Phần này đưa ra khảo sát để thấy được tầm ảnh hưởng của cả 2 khi training data bị cố định.
    Họ sử dụng Iresnet100 làm backbone
    Tích của số người và hình ảnh của mỗi người được cố định ở mức 80k trong mọi thiết lập. Vì vậy, bốn thiết lập có thể được biểu thị là 1w 80, 2w 40, 4w 20 và 8w 10. Ví dụ, 1w 80 có nghĩa là 10k người và mỗi id chứa 80 hình ảnh.

Kết quả: ![](images/5.2.%20Result.png)
    Có thể thấy 4w_20 cho kết quả tốt nhất.

## 5.3. Long tail distribution

Purity (độ tinh khiết) và long-tail distribution (đuôi dài, trong cả dataset chỉ có 1 vài IDs thực sự là depth data) là các yếu tố thiết yếu ảnh hưởng đến hiệu suất của các mô hình FR hiện đại.

Điều kiện thực nghiệm
    Dataset: Tiến hành làm sạch dư liệu bằng mô hình ... để thu được dataset sạch hơn. Lặp lại quá trình này với nhiều mô hình để thu được dataset sau cùng là WebFace35M. Họ tiếp tục lọc ra các IDs có số ảnh nhỏ hơn 10.
    Backbone: Iresnet100 và thêm lần lượt 0, 25, 50, 100% dữ liệu đuôi dài.
Kết qủa:

    ![](images/5.3.%20Result.png)

# 6. Dataset and Comparison Results

## 6.1. Training datasets
Phần này nói về các dataset được sử dụng phổ biến trong lĩnh vực FR.

![](images/6.1.%20Training%20sets.png)

## 6.2. Testing datasets và Metrics
Các metric được sử dụng phổ biến trong lĩnh vực FR là:
1. Verification accuracy: testset có 2 hoặc nhiều khuôn mặt của 1 người hoặc khác người (hiển nhiên các hình ảnh này sẽ có cùng IDs). Các ảnh này sẽ được đưa cho model để nó dự đoán xem nó có cùng 1 người hay không.
    LFW (Labelled Faces in the Wild) là tập phổ biến để đánh giá "verification accuracy" với 6000 cặp ảnh khuôn mặt. Các phiên bản khác của LFW như CALFW và CPLFW cũng được sử dụng.
    CFP-FP là tập khác với ảnh khuôn mặt chụp từ chính diện và góc nghiêng.
    YTF (YouTube Faces) cung cấp 5000 cặp video để đánh giá độ chính xác trong xác thực.
    Table sau đánh giá verification accuracy (hiệu suất xác thực) của các mô hình a state of art

    ![](images/6.2.%20Verification%20accuracy.png)

2. MegaFace dataset
    Đây là 1 tập dữ liệu tiêu chuẩn (benchmark) dùng để đánh giá các mô hình FR. Nó gồm 2 tập nhỏ.
        Gallery set: chứa hơn 1 triệu image của 690k người khác nhau và là các ảnh đã biết danh tính và được lưu trữ trong hệ thống FR.
            Khi nhận diện khuôn mặt, mô hình sẽ só sánh đầu vào với các ảnh trong gallery set để tìm kiếm hoặc xác thưucj danh tính. Gallery set của megaface chứa hàng triệu ảnh khuôn mặt tạo nên thử thách lớn cho các hệ thống nhận diện.
        Probe set:
            Là tập hợp các ảnh chưa biết danh tính, thường được dùng làm testset để nhận diện hoặc xác thực để đối chiếu với các ảnh trong gallery set.
            Gồm 2 tập set là Facescrub và FGNet dùng để đánh giá mô hình trong các tình huống khác nhau và các ảnh đều là ảnh chưa biết danh tính
    MegaFace có rất nhiều kịch bản để đánh giá hệ thống FR bao gồm:
        Xác thực và nhận diện khuôn mặt trong tư thế bất biến (pose invariance) với large và small dataset. (small nếu nó được huấn luyện dưới 0,5 M ảnh).
        Với face identification, các đường CNC và ROC được sử dụng để đánh giá
        Với face verification, 'Rank-1 Acc' và 'Ver' sẽ được đánh giá. (chi tiết đọc tài liệu)

3. IJB - A: 1 phương thức testing khác.
    Với face verification: true accept rate (TAR) và false positive rates (FAR) được reported.
    Với face identification: true positive identi cation rate (TPIR) và false positive identi cation rate (TPIR) and the Rank-N accuracy được báo cáo.
        Gần đây ít research published kết quả của họ lên IJB-A vì họ đã đạt được hiệu suất cao trên phương pháp tính điểm này.
        Bên cạnh IJB-A còn có các phương thức IJB-B và IJB-C, **kết quả các phương pháp FR theo phương thức đánh giá IJB-B và IJB-C được hiện dưới bảng sau**.

        ![](images/6.2.%20IJB%20protocol.png)

4. Các protocol được sử dụng để đánh giá hệ thống FR trên không có bất kỳ hạn chế nào về mặt thời gian. Đối với giao thức Face Recognition Under Inference Time conStraint (FRUITS) thì không.
    FRUITS được thiết kế để đánh giá toàn diện hệ thống FR (hình như chỉ có đánh giá xác thực khuôn mặt) với giới hạn thời gian.
    FRUITS-x (x có thể là 10, 50, 100) đánh giá hệ thống FR phải xác thực trong vòng x mili giây bao gồm cả các bước detect và căn chỉnh.
    FRUITS-100 nhằm đến việc đánh giá hệ thống FR có thể triển khai trên các thiết bị di động, FRUITS-500 nhằm đánh giá các mạng hiện đại và phổ biến được triển khai trong hệ thống giám sát cục bộ. FRUITS-1000 nhàm đánh giá khả năng triển khai trên các cloud.
    **Bài báo cung cấp table các phương pháp FR đánh giá theo phương thức này.**

**Các metric sẽ được giới thiệu ở phần 8.**

# 7. Applications
Phần này nói về 1 số ứng dụng phổ biến của FR như face clustering (phân cụm khuôn mặt), atribute recognition (nhận diện thuộc tính khuôn mặt) và face generation

## 7.1. Face clustering
Từ 1 collection face image chưa từng nhìn thấy. Model sẽ tiến hành phân cụm nhóm các hình ảnh mà nó cho là của 1 người lại với nhau thành 1 cụm.
Ứng dụng trong công nghiệp
- Phân loại khuôn mặt trong album ảnh
- Tóm tắt các nhân vật trong video

Face clustering sử dụng các embedding của khuôn mặt được tạo ra từ 1 hệ thống FR đã được huấn luyện tốt. Embedding càng chất lượng thì càng cải thiện khả năng phân cụm.
2 phương pháp chính trong face clustering
- Unsupervised: Xem mỗi embedding là 1 điểm trong feature space và sử dụng các thuật toán phân cụm học không giám sát phổ biến như K-means (yêu cầu cluster có hình dạng lồi), spectral clustering (yêu cầu các cụm có số lượng điểm tương đối bằng nhau), DBSCAN (giả định các cụm có cùng mật độ)
- Các phương pháp dựa trên GCN (Graph Convolutional Network): Các phương pháp dựa trên GCN là có giám sát, thường đạt hiệu suất tốt  hơn với các thuật toán unsupervised vì chúng có thể học cách nhóm các feature với sự hộ trợ của label.

**Phần này giới thiệu 1 vài phương pháp GCN được publish gần đây. Để biết về nó cần làm rõ 1 vài khái niệm**
- Từ 1 face dataset, ta sử dụng 1 mô hình CNN đã trained để trích xuất tất cả feature của các image ta thu được 1 sets các feature của các image như sau.

    ![](images/7.1.%20Feature%20Image.png)

    Với n là số lượng ảnh trong face dataset, n là số chiều của feature.

- Sau đó với mỗi feature trong tập trên ta coi nó là 1 đỉnh trong feature space và sử dụng cosine similarity để tìm k-nearest neightbors với mỗi sample. 1 biểu đồ graph sẽ được dụng lên để thể hiện mối tương quan giữa các feature G=(V,E) => Được biểu diễn trong máy tính bởi 1 ma trận kề A.

1. Yang et al. [194] đề xuất 1 framework dựa trên GCN bao gồm 3 module là 
    - proposql generator (trình tạo đề xuất - tức là đồ thị con có khả năng là cluster) từ đồ thị tổng quát ma trận kề A.
        Để thực hiện được điều này, họ loại bỏ các edges có giá trị nity dưới threshold và kích thước của đồ thị con phải nhỏ hơn 1 giá trị maximum nào đó.

    - GCN-D (Phát hiện cụm): Đầu vào là 1 cluster P được proposql generator đề xuất. Nó đánh giá khả năng đề xuất tạo thành 1 cluster mong muốn bằng cách 2 metric IoU (độ gần của cụm đề xuất với cụm thực sự) và IoP (dô độ thuần khiết của cụm). GCN-D được huấn luyện để dự đoác cả 2 metric này vằng MSR

    - GCN-S (Phân đoạn cụm): có cấu trúc tương tự như GCN-D, thực hiện phân đoạn các cụm để tinh chỉnh và cải thiện cụm đề xuất ban đầu.

2. Wang et al. [195] cũng đề xuất giải pháp phân cụm khuôn mặt dựa trên GCN nhưng thay vì dự đoán tính liên kết giữa các cụm như trong phương pháp trước, phương pháp này tập trung vào việc dự đoán mức độ tương tự (similarity) giữa hai đặc trưng.
    Instance Pivot Subgraphs (IPS): là các đồ thị con (subgraph) được xây dựng quanh một điểm trung tâm (pivot) đại diện cho mỗi điểm ảnh 𝑝 trong đồ thị 𝐺.
    Mỗi IPS bao gồm các nút lân cận gần nhất của 𝑝 (K-Nearest Neighbors - KNN) và các lân cận bậc cao (đến tối đa 2 bước nhảy - 2-hop neighbors) của 𝑝.
    ....

## 7.2.1. Face attribute recognition.
DỰ đoán các thuộc tính của khuôn mặt là 1 trong những ứng dụng rộng rãi của face embedding. Bằng cách trích xuất các đặc điểm từ face images, mạng neutron có thể ước tính age, gender, expression (biểu cảm), hairstyle và các thuộc tính khác của khuôn mặt. Phần lớn, việc nhận diện các thuộc tính dựa trên localization results đã được tóm tắt ở phần 3.1.

Để dự đoán, multi-task learning được sử dụng rộng dãi để nhận dạng 1 nhóm các thuộc tính cùng 1 lúc.

1.  Liu et al. [168] đề xuất mô hình Anet để trích xuất các đặc điểm khuôn mặt và sử dụng nhiều bộ phân loại máy vector ( multiple support vector machine (SVM)) để dự đoán 40 thuộc tính khuôn mặt, và sau đó được điều chỉnh ne-tuned (fine-tune) bởi nhiều attribute tags.
    Trong bước ne-tined stage, nhiều patches của face được tạo ra với mỗi image và 1 giải pháp trích xuất feature nhanh gọi là interweaved operation (hoạt động đan xen) được đề xuất để phân tích các patches này.

    Chi tiết mô hình Anet đọc doc:

    ![](images/7.2.%20Anet%20model..png)

2. **PS-MCNN: (Partially Shared Multi-task Convolutional Neural Network)** là 1 mô hình mạng neutron tích chập đa nhiệm được đề xuất bởi Cao và các cộng sự nhằm dự đoán các thuộc tính khuôn mặt. 
    Gồm 2 thành phần chính:
        SNet(shared network): Mạng này dùng để trích xuất các đặc trưng chung giữa các nhiệm vụ và chia sẻ thông tin giữa các nhánh (branches) khác nhau.
        TSNet (Task Specific Network): Mạng dành riêng cho từng nhiệm vụ, mỗi TSNet sẽ xử lý các thuộc tính của một nhóm nhất định.

    Trong PS-MCNN, các thuộc tính khuôn mặt được chia thành bốn nhóm dựa trên vị trí của chúng trên khuôn mặt: phần trên, phần giữa, phần dưới, và toàn bộ khuôn mặt.
    Để cải thiện hơn nữa, Cao và cộng sự đã phát triển một mô hình mới tên là PS-MCNN-LC (Partially Shared Network with Local Constraint). Trong mô hình này, họ bổ sung thêm một hàm mất mát mới tên là LCLoss để tận dụng thông tin nhận dạng (identity information) của các khuôn mặt. Hàm mất mát LCLoss giúp mô hình học cách nhận ra sự tương đồng giữa các thuộc tính khuôn mặt từ cùng một danh tính, bằng cách ràng buộc các đặc trưng của các khuôn mặt cùng danh tính lại gần nhau hơn trong không gian đặc trưng.

3. Bên cạnh multi-task learning, các đặc trưng như tuổi, ... có thể được xem như single-task learning. 
    Thay vì sử dụng mạng neutron học sâu phức tạp, Zhang et al đề xuất 1 mạng neutron tên là c3AE để dự đoán tuổi của 1 người. Nó chỉ gồm 5 convolution layes và 2 dense layers.
    Input của model là ảnh RGB chứa mặt đã được cắt bởi các phương pháp face aligment trước đó.
    Chi tiết mô hình C3AN: ![](images/7.3.%20C3AE.png)

    Đối với expression recognition (nhận dạng biểu cảm). Do có 1 lượng lớn đa dạng các biểu cảm trên khuôn mặt và các biểu cảm này sẽ biến đổi lớn do các đặc điểm nhân khẩu học khác nhau.

    Phương pháp Deviation Learning Network (DLN) của Zhang và cộng sự được thiết kế để loại bỏ đặc điểm nhận dạng của khuôn mặt (identity attributes) khỏi đầu vào. DLN bao gồm hai mô hình:

        Mô hình nhận dạng (identity model) và
        Mô hình khuôn mặt (face model),
        Cả hai đều dựa trên mô hình Inception-ResNet FaceNet đã được huấn luyện trước. Tuy nhiên, chỉ các tham số của face model là có thể điều chỉnh trong quá trình huấn luyện, còn tham số của identity model thì được cố định.

        Cụ thể:

        Vface và Vid là hai vector đầu ra từ face model và identity model, có kích thước 512 chiều.
        Vector biểu cảm (expression vector) được tính bằng hiệu (Vface - Vid) nhằm loại bỏ các đặc điểm nhận dạng cá nhân khỏi khuôn mặt, chỉ giữ lại thông tin biểu cảm.
        Vector này sau đó được chuyển thành không gian 16 chiều thông qua một mô-đun bậc cao (high-order module) để làm nổi bật đặc điểm biểu cảm.
        Cuối cùng, việc dự đoán biểu cảm được thực hiện dựa trên các đặc trưng trong không gian 16 chiều này, thông qua một crowd layer – lớp này có chức năng giảm thiểu sự sai lệch trong việc gán nhãn biểu cảm, giúp mô hình đưa ra dự đoán chính xác hơn.