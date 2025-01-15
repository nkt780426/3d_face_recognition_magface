# 4. Algorithms

Phần này giới thiệu về các FR algorithms trong những năm gần đây. Dựa trên nhiều khía cạnh của các FR modeling, họ chia phần này thành các mục sau.
1. designing loss function
2. refining embedding
3. FR with massive IDs
4. FR on uncommon images
5. FR pipeline acceleration
6. close-set training 

## 4.1. Loss function
Tổng kết: các phương pháp ở phần 4.1.1 cho ra hiệu xuất thấp hơn so với các phương  pháp ở phần 4.1.2
Phương pháp mạnh nhất là MagFace và AdaFace.

### 4.1.1. Loss based on metric learning

Mục tiêu của phần này là giới thiệu các hàm loss function được sử dụng trong lĩnh vực FR. Các loss function này chính là 1 metric trong quá trình training (loss bassed on metric learning là vì vậy). Bản chất model học dựa theo loss function là quá trình model thiết lập 1 feature face, sao cho nếu 2 face thuộc cùng 1 id khi tham chiếu lên feature face này, khoảng cách euclidean/khoảng cách cosin của chúng phải là nhỏ nhất có thể.

1. Với pipeline của face verification (chỉ cần xác định true hay fale), *Han và các công sự của a*[40] sử dụng **cross-entropy** làm loss function trong quá trình train model.
    ![](images/4.1.1.%20cross-entropy.png)
    
    yij là nhãn nhị phân xác định xem 2 image i và j có cùng 1 ID hay không (yij=1 nếu cùng id và = 0 nếu ngược ID).
    pij là giá trị logits (đầu ra của mô hình sau khi áp dụng hàm sigmoid) biểu thị xác xuất mà mô hình dự đoán 2 khuôn mặt của image xem có cùng danh tính hay không.
    Mục tiêu của hàm loss này là đo lường sự khác biệt giữa pij (gía trị dự đoán) và yij (giá trị gốc của label). Khi pij càng gần yij thì loss càng nhỏ và ngược lại => Model biết ma trận trọng số trong epoch như thế nào với epoch trước đó mà cải thiện.

2. **contrastive loss**
    Khác với [40], contrastive loss được [41] đề xuất để so sánh trực tiếp feature của 2 face trong 2 image. Vẫn là tiêu chí: nếu 2 face thuộc cùng 1 ID thì khoảng cách giữa 2 feature extract được từ 2 face trên tham chiếu lên traind space phải gần nhau nhất và ngược lại. Loss function contrastive như sau.

    ![](images/4.1.1.%20contrastive.png)

    i, j là 2 ảnh chứa 2 khuôn mặt trong dataset và fi, fj là feature mà model extract được từ chúng => ||fi-fj||^2 là khoảng cách euclidea của chúng.
    m là margin (biên độ) để mở rộng khoảng cách của các mẫu có ID khác nhau (negative pairs)

    Công thức trên được tính toán dựa theo khoảng cách euclidean, ta có thể thay thế nó nếu sử dụng khoảng cách cosin.
    
    ![](images/4.1.1.%20contrastive-cosin.png)

    trong d là cosin simalarity giữa 2 feature fi và fj, w và b là 2 tham số mà model có thể học và tỷ lệ có thể học được. phi là sigmoid fuction.

3. **BioMetricNet** [42]
    Khác với 2 hàm loss trên, có thể thấy model phụ thuộc vào 1 công thức loss xác định. Tuy nhiên BioMetricNet không sử dụng metric cố định như euclid hay cosin để so sánh đặc trưng khuôn mặt. Thay vào đó, BioMetricNet quyết định training space bằng cách học 1 biểu diễn tiềm ẩn (latent representation) mà các positive pair và negative pair được map vào các phân phối mục tiêu rõ ràng và tách biệt.

    Quá trình hoạt động của BioMetricNet bao gồm: trích xuất đặc trưng của face pair, ánh xạ nó vào không gian mới nơi quyết định 2 face có cùng ID hay không được tạo ra. Loss của BioMetricNet rất phức tạp dùng để đo lường sự phân phối xác suất thống kê của các positive pair và negative pair trong không gian này.

4. **FaceNet-Triplet loss** [34]
    FaceNet đề xuất 1 **triplet loss**: Đầu vào gồm 3 ảnh, anchor và positive thuộc cùng 1 ID và negative khác ID. 
    Mục tiêu của triplet loss là **để tối thiểu hóa khoảng cách giữa anchor và positive và tối đa khoảng cách giữa ảnh anchor và negative**.
    Động lực ra đời của hàm này là [44] trong bối cảnh cần phân loại nearest-neighbor. Triplet loss đảm bảo các ảnh x^a (anchor) của 1 người sẽ gần với tất cả các image x^p (positive) của người  đó trong dataset và xa hơn với bất kỳ x^n (negative) của bất kỳ người nào khác.

    ![](images/4.1.1.%20triplet.png)

    T có thể hiểu là dataset sao cho đảm bảo các điều kiện hình thành anchor, positive, negative 
    f(x): face embededing của x, 
    alpha: là margin (biên độ). Cụ thể ta muốn đảm bảo khoảng cách (anchor, posivte) luôn nhỏ hơn khoảng cách (anchor, negative) với 1 biên độ nhất định margin. Đây là 1 tham số cố định được chọn trước khi train model.
    Loss sẽ là 0 nếu tất cả bộ ba thuộc T đảm bảo điều trên.

    Để đảm bảo qúa trình train hội tụ nhanh (fast convergence), ta cần phải nhanh chóng chọn ra bộ 3 vi phạm triplet loss, cùng nhìn lại ý nghĩa của triplet loss.
    
    ![](images/4.1.1.%20triplet%20purpose.png)

    Có thể thấy triplet loss rất nhạy cảm với việc chọn bộ 3 (xa,xp,xn). Thay vì thử bừa positve và negative, ta cố gắng chọn xp, xn sao cho hard-positive và hard-negative xảy ra
        hard-positive: khoảng cách (xp, xa) lớn nhất => Giúp mô hình học cách thu nhỏ khoảng cách giữa anchor và positive.
        hard-negative: khoảng cách (xn,xa) nhỏ nhất => Nhằm tăng độ khó cho mô hình khi cố gắng phân biệt anchor với negative.

5. Kang loss (tự đặt)
    Kang và các cộng sự của anh ấy[44] đã đơn giản hóa contrastive và triplet loss và thiết kế 1 công thức loss mới.

    ![](images/4.1.1.%20kang%20loss.png)

    trong đó ![](images/4.1.1.%20kang%20loss%20explain.png)

    Tổng hợp lại, hàm mất mát này giúp mô hình học nhanh và hiệu quả hơn bằng cách tập trung vào các ví dụ khó (hard examples) và cải thiện sự phân tách giữa các danh tính khác nhau trong không gian đặc trưng.

Tổng hợp lại: Contrastive loss dùng để đo khoảng cách giữa các image pair, triplet loss đảm bảo mối quan hệ giữa bộ 3 sample (anchor, positive và negative). Ngoài ra còn rất nhiều loss thuộ dạng metric learning được các researcher phát triển bằng cách đưa nhiều sample hơn trong mô tả hàm loss.

6. center loss
    Dung để giảm thiểu sự phân tán của các feature trong cùng 1 class bằng cách đặt chúng về gần trung tâm của class. Center loss thường được sử dụng với softmax để tăng tính phân biệt giữa các ID (refine - tinh chỉnh). Công thức center loss như sau.

        ![](images/4.1.1.%20center.png)

    Lsoftmax dựa trên label của training data, i là image dùng để train với nhãn thật là yi, xi là deep feature, m là số lượng training class, cij biểu thị trung tâm của class chứa ảnh yi

### 4.1.2. Larger margin loss
Bài toán FR có thể được coi là classification problem và sử dụng softmax để train model. Larger margin loss (hay tổn thất dựa trên biên độ góc - angular margin bassed loss) được lấy cảm hứng từ hàm softmax. Dạng loss func này được sử dụng phổ biến trong lĩnh vực FR trong học tập và công nghiệp. Nó cải thiện đáng kể hiệu xuất của mô hình FR.

Bản chất ta vẫn muốn cacs facial images (đặc tính khuôn mặt) của các mặt có cùng ID sẽ gần nhau hơn trong không gian train và các ảnh khác ID sẽ xa nhau trong không gian train. Kết quả large margin loss khuyến khích sự đồng nhất trong 1 lớp (intra-class compactness) và trừng phạt sự đồng nhất của các ảnh khác ID - hay sự khác biệt giữa các lớp (inter-class separability).

Công thức softmax, các khái niệm sau đây sẽ được sử dụng  ở toàn bộ phần này:
    ![](images/4.1.2.%20softmax.png)

    W là ma trận trọng số thu được sau last fully connected layer và được sử dụng tính loss trong quá trình lan truyền ngược.
    Wj đại diện cho weight của class j
    xi, yi: là face embedding của ảnh i và group id thực tế của ảnh i. Trong class yi, ảnh xi có vai trò positive và với class yj (j#i) thì ảnh xi có vai trò negative. 

**Softmax truyền thống chỉ đơn thuần xác định nhãn cho mỗi mẫu mà không tối ưu hóa khoảng cách giữa các danh tính trong không gian biểu diễn. Tuy nhiên nó là nền tảng để mở rộng các hàm loss ở phaafnnafy**

Dựa trên positive sample và negative sample, softmax có thể được viết lại thành.
    ![](images/4.1.2.%20softmax-positve,negative.png)

1. **L-softmax** [46] đây là loss function đầu tiên thiết kế theo margin bassed loss bằng cách đo features angles (góc của các đặc điểm).
    Đầu tiên nó bỏ qua bias của mỗi class bj và chuyển tích Wj * xi thành ‖Wj‖·‖xi‖·cos(θj) trong đó θj là góc giữa xi và ma trận trọng số Wj của class j.
    Để mở rộng biên độ góc giữa các lớp, L-softmax biến đổi cos(θyi) thành ψ(θyi) bằng cách thu hẹp khoảng cách boundary (viền ngoài class) với trung tâm class (center class)

    ![](images/4.1.2.%20L-softmax.png)

    m là 1 tham số integer cố định , góc θ được chia thành m đoạn  [kπ/m , (k+1)π/m] ....

2. **A-softmax**: Được giới thiệu trong Sphereface [47]
    Là bản cải tiến L-softmax bằng cách chuẩn hóa trọng số của từng class (Wj) trước khi tính loss  và kết hợp với L-softmax để tăng độ chính xác khi train. Loss trở thành

    ![](images/4.1.2.%20A-softmax.png)

    Trong quá trình train mạnh sphereface, thực tế loss lại là: (1-α)*softmax + α*A-softmax với α thuộc đoạn [0,1] và thay đổi trong quá trình train. Thiết kế này có 2 mục đích
        Nếu sử dụng A-softmax làm loss function trực tiếp sẽ dẫn đến hard convergence (hội tụ cứng) vì nó đẩy mạnh các feature của các IDs khác ra xa
        Nếu train với softmax trước sẽ giảm angle θyi giữa feature i và trọng số Wyi liên quan đến nó. => Cos(mθ) tribg A-softmax sẽ nằm trong 1 vùng đơn điệu và dễ dàng để có được 1 loss thấp trong quá trình gradient decending.

3. **Normface loss**
    Bên cạnh việc normalize weight của mỗi class, NormFace [48] đề xuất normalize cả face embedding, sau đó scale up normalized face embedding bằng 1 tham số tỷ lệ s. Điều này sẽ làm giảm vấn đề mất cân bằng dữ liệu giữa positive và negative sample và có sự hộ tụ tốt hơn. Tuy nhiên Normface không sử dụng margin như Sphereface đối với positive sample. Công thức NormFace loss như sau:
    
    ![](images/4.1.2.%20Normface.png)

    W~và x~ ở trên là normalized class và normalized face embedding
    s là tham số scale up (s>1)

    **Điều này giúp các đặc trưng khuôn mặt được phân bố đều trên một hypersphere, làm tăng khả năng phân biệt của mô hình mà không cần thêm margin.**

4. **AM-softmax[49]** và **CosFace[50]**
    Họ mở rộng hàm softmax cơ bản bằng cách trừ đi margin m bên ngoài cos(θ). Công thức như sau
    ![](images/4.1.2.%20cosface.png)

5. **ArcFace[51]**
    Đưa margin vào góc trong cos(θ) và làm margin dễ lý giải hơn

    ![](images/4.1.2.%20arcface.png)

6. P2SGrad [52]
    **Các tổn thất dựa trên angular margiin bassed loss ở trên (CosFace và ArcFace) bao gồm các hyper-parameters nhạy cảm. Điều này có thể làm cho quá trình train không ổn định. Bản chất các hàm CosFace và ArcFace là các hàm mất mát thêm margin góc cho các positive sample để tăng cường khả năng phân biệt. Tuy nhiên chúng có 2 hạn chế.**
        Nhạy cảm với các siêu tham số: CosFace và ArcFace cần các tham số như margin 𝑚 và scale s, điều này có thể làm cho quá trình huấn luyện không ổn định và khó hội tụ.
        Chỉ tác động lên mẫu dương: Cả CosFace và ArcFace chỉ thêm margin cho các mẫu dương, do đó chỉ đảm bảo rằng các mẫu dương sẽ gần với trung tâm lớp của chúng mà không tối ưu hóa khoảng cách giữa các mẫu âm (negative samples) và lớp dương.

    P2SGrad [52] được đề xuất để giải quyết thách thức này bằng cách trực tiếp thiết kê các gradient cho các mẫu theo cách thích ứng. Thay vì sử dụng các margin cố định, P2SGrad điều chỉnh gradient dựa trên các mẫu đầu vào, giúp mô hình linh hoạt hơn trong việc học và tăng khả năng hội tụ.

7. SV-Softmax[53] và các biến thể
    Đây là 1 cải tiến khác so với AM-softmax, CosFace và Arcface. Nó không chỉ tập trung vào positive samples mà còn tác động đến negative samples.
    Thay vì chỉ đưa các mẫu dương lại gần trung tâm lớp, SV-Softmax đẩy các mẫu âm khó (hard negative samples) ra xa trung tâm lớp của các mẫu dương.
    Cách này đảm bảo rằng các lớp sẽ tách biệt hơn, giảm thiểu việc các mẫu âm xâm nhập vào không gian của lớp dương, từ đó tăng khả năng phân biệt của mô hình.
    Để chọn các mẫu âm khó, SV-Softmax sử dụng nhãn nhị phân 𝐼𝑗, chỉ định xem một mẫu 𝑗 có phải là hard negative không, dựa vào điều kiện:

    ![](images/4.1.2.%20L-softmax.png)
    
    SV-X-Softmax là phiên bản mở rộng của SV-Softmax bằng cách bổ sung large margin cho các mẫu dương. Hàm mất mát được điều chỉnh như sau:
    
    ![](images/4.1.2.%20SV-X-softmax.png)

    MV-Softmax là bước tiến hóa tiếp theo của SV-Softmax, trong đó hàm ℎ(𝑡,𝜃𝑗,𝐼𝑗) được định nghĩa lại để cải thiện khả năng điều chỉnh trọng số của các mẫu âm. Thay đổi này giúp MV-Softmax linh hoạt hơn trong việc xác định và xử lý các mẫu âm khó, tăng độ chính xác và ổn định trong quá trình huấn luyện.

    Tóm lại: SV-Softmax và các biến thể của nó như SV-X-Softmax và MV-Softmax cải thiện sự phân biệt của mô hình bằng cách tối ưu hóa các mẫu âm khó và tăng khoảng cách giữa các lớp dương và âm, giúp mô hình học cách phân loại chính xác hơn.

8. Ring loss[55]
    CosFace [50] and ArcFace [51] thực hiện chuẩn hóa loss. Ring Loss là một hàm mất mát đặc biệt nhằm chuẩn hóa độ dài của các embedding đặc trưng khuôn mặt về một giá trị cố định 𝑅. Điều này giúp các embedding có cùng độ dài, tăng độ ổn định và giúp mô hình hội tụ tốt hơn khi huấn luyện. Công thức Ring Loss:
        ![](images/4.1.2.%20Ringloss.png)
        m là batch size
        λ là trọng số để đánh đổi giữa hàm mất mát chính. Trong [55], hàm mất mát chính được đặt thành softmax và SphereFace [47].
    
9. **Trong trường hợp training set có các ảnh bị noise**, Hu et al[56] đề xuất phương pháp học với dữ liệu nhiễu, trong đó **góc 𝜃 giữa mẫu và trung tâm lớp tương ứng đóng vai trò đánh giá độ nhiễu của mẫu**. Mẫu có góc 𝜃 nhỏ hơn sẽ ít nhiễu hơn và được gán trọng số cao hơn trong quá trình huấn luyện để mô hình ưu tiên học từ chúng. 
    Phân phối của góc 𝜃:
        Trong một tập training set chứa nhiễu, góc 𝜃 thường có dạng phân phối Gaussian với hai cực trị. Hai cực trị này tương ứng với các mẫu nhiễu và mẫu sạch.
        Phân phối của 𝜃 có các điểm cực trái (𝛿𝑙) và cực phải (𝛿𝑟), cùng với các đỉnh phân phối Gaussian ở 𝜇𝑙 và 𝜇𝑟 (nếu chỉ có một Gaussian, 𝜇𝑙=𝜇𝑟).
    
    ![](images/4.1.2.%20Ringloss%20AM-softmax.png)

    ![](images/4.1.2.%20Ringloss%20AM-softmax-2.png)

10. Sub-center ArcFace và CurricularFace: hai phương pháp cải tiến của ArcFace nhằm giải quyết vấn đề dữ liệu nhiễu và tối ưu hóa hiệu suất mô hình nhận diện khuôn mặt.
    ![](images/4.1.2.%20Sub-center%20Arceface.png)

    ![](images/4.1.2.%20CuricularFace.png)

    alpha là momemtem 

11. NPCFace[59]:
    NPCface – một kỹ thuật sử dụng trong huấn luyện các mô hình nhận diện khuôn mặt để xử lý các trường hợp khó (hard cases) một cách hiệu quả. Ý tưởng chính của NPCface là tập trung vào những mẫu dữ liệu khó để cải thiện độ chính xác của mô hình.

    Mục tiêu của NPCface là nhấn mạnh việc huấn luyện trên các mẫu khó bằng cách điều chỉnh margin dựa trên độ khó của các mẫu, từ đó cải thiện khả năng phân loại của mô hình trong các tập dữ liệu lớn và đa dạng.

    ![](images/4.1.2.%20NPCface.png)

12. UniformFace[60] nhằm tối ưu hóa sự phân bố của các lớp trên manifold (đa tạp) dạng hypersphere để đạt sự đồng đều và tối đa hóa khoảng cách giữa các lớp, giúp cải thiện hiệu quả phân biệt giữa các lớp khuôn mặt.

    UniformFace nhận xét rằng các hàm large margin loss như CosFace và ArcFace chưa cân nhắc đến sự phân bố của tất cả các lớp. Các lớp có thể không được phân bố đều trên manifold, dẫn đến việc mô hình khó khăn khi phân biệt các lớp tương tự nhau.

    Với quan điểm rằng các đặc trưng khuôn mặt nằm trên manifold dạng hypersphere, UniformFace áp dụng ràng buộc equidistributed (phân bố đều), cố gắng tối đa hóa khoảng cách tối thiểu giữa các center (tâm) của các lớp, giúp tận dụng tối đa không gian đặc trưng.

    ![](images/4.1.2.%20UniformFace'.png)

13. RegularFace[61]
    Zhao et al. (RegularFace) mở rộng khái niệm “inter-class separability” (khả năng tách biệt giữa các lớp) vào các hàm loss dựa trên large-margin để cải thiện độ chính xác. RegularFace đo lường khoảng cách góc (cosine similarity) lớn nhất giữa một lớp 𝑖 và các lớp khác với công thức:

    ![](images/4.1.2.%20RegularFace.png)

14. Variational Prototype Learning (VPL)
    VPL mở rộng việc đo lường khoảng cách từ mẫu tới tâm lớp (sample-to-prototype) bằng cách sử dụng khoảng cách từ mẫu đến prototype biến thiên (variational prototype), giúp cải thiện tính linh hoạt của mô hình.

    ![](images/4.1.2%20Variational%20Prototype%20Learning.png)

15. UIR (Unlabeled ID Regularization)
    UIR huấn luyện bộ trích xuất đặc trưng trong môi trường bán giám sát, bằng cách đưa thêm các dữ liệu không nhãn vào huấn luyện. => **Không dùng**

    ![](images/4.1.2.%20URI.png)

Các phương pháp RegularFace, VPL, và UIR đều nhắm đến việc tối ưu hóa khả năng tách biệt giữa các lớp trong không gian đặc trưng, nhưng mỗi phương pháp lại sử dụng các chiến lược khác nhau để đạt được điều này. RegularFace tập trung vào việc làm đều các khoảng cách giữa các lớp, VPL mở rộng việc đo lường khoảng cách với prototype biến thiên, và UIR tận dụng dữ liệu không nhãn để cải thiện phân bố của các lớp.

16. AdaCos
    AdaCos là một biến thể của CosFace, có khả năng tự động điều chỉnh tham số scale 𝑠 trong quá trình huấn luyện. AdaCos nhận thấy rằng nếu 𝑠 quá nhỏ, xác suất của một mẫu sẽ thấp, ngay cả khi khoảng cách góc giữa mẫu và tâm lớp của nó là nhỏ. Ngược lại, nếu 𝑠 quá lớn, xác suất sẽ đạt gần 1 dù khoảng cách góc lớn.

    ![](images/4.1.2.%20Adacos.png)

17. FairCos
    Fair Loss cũng điều chỉnh tham số margin 𝑚 trong quá trình huấn luyện, nhưng sử dụng học tăng cường (reinforcement learning). Giá trị của 𝑚 được điều chỉnh theo trạng thái của quá trình huấn luyện, nhằm tối ưu hóa khả năng phân biệt giữa các lớp.

    ![](images/4.1.2.Fair%20Loss.png)
    
18. AdaptiveFace (AdaM-Softmax)
    AdaptiveFace giải quyết vấn đề mất cân bằng dữ liệu giữa các lớp khác nhau, khi mà một số ID có ít mẫu và số khác có nhiều mẫu. Thay vì sử dụng margin cố định cho mọi lớp, AdaptiveFace điều chỉnh margin 𝑚 dựa trên số lượng mẫu của từng lớp.

    ![](images/4.1.2.%20AdaptiveFace-2.png)

    ![](images/4.1.2.%20AdaptiveFace.png)

Đoạn văn sau thảo luận về các cải tiến trong hàm loss để tối ưu hóa quá trình học trong nhận diện khuôn mặt, đặc biệt là việc xử lý các mẫu khó (hard samples) và cải thiện chất lượng của biểu diễn đặc trưng. Dưới đây là tóm tắt cùng với các công thức liên quan:

19. Distribution Distillation Loss (DDL)
    Huang et al. nhận thấy rằng các hàm loss sử dụng margin lớn thường gặp khó khăn với các mẫu khó. Họ sử dụng ArcFace để tạo ra một phân phối từ các mẫu dễ (teacher) và một phân phối từ các mẫu khó (student). DDL được đề xuất để buộc phân phối của mẫu khó gần với phân phối mẫu dễ

    ![](images/4.1.2.%20DDL.png)

20. MagFace
    MagFace giới thiệu một cơ chế học để phân phối đặc trưng trong lớp tốt hơn bằng cách kéo các mẫu dễ gần tâm lớp với độ lớn lớn hơn, trong khi đẩy các mẫu khó ra xa với độ lớn nhỏ hơn.

    ![](images/4.1.2.%20MagFace.png)

21. **AdaFace**: Thằng loss mạnh nhất trong bài research này đề cập
    Ý tưởng chính: AdaFace điều chỉnh gradient trong quá trình lan truyền ngược dựa trên chất lượng của hình ảnh. Các mẫu khó được nhấn mạnh khi chất lượng hình ảnh cao, và ngược lại.

    ![](images/4.1.2.%20AdaFace.png)

### 4.1.3. FR in unbalanced training data

Trong 1 Large-scale face dataset, không thể tránh khỏi việc có 1 số lớp có lượng dữ liệu không cân bằng. Những feature thuộc các class này (không chiếm ưu thế - non-dominate IDs) sẽ được nén vào 1 khu vực trên hypershere. Điều này dẫn đến 1 số vấn đề trong quá trình training. Do đó, đối với các hiện tượng mất cân bằng dữ liệu khác nhau, các phương pháp khác nhau đã được đề xuất.

1. **Long tail distributed - Phân phối đuôi dài**:
    Là hiện tượng mất cân bằng dữ liệu đầu tiên, tồn tại rộng rãi trên đa số các mainstream training set (tập dữ liệu đào tạo chính thống) như MS-Celeb-1M.

    Trong MS-Celeb-1M, hầu như tất cả các IDs đều có số lượng ảnh ít và chỉ có phần nhỏ số IDs có lượng lớn ảnh.

    Zhang et al. [73] đã tạo ra rất nhiều experiment để chứng minh rằng với tất cả tail data trong quá trình training không thể giúp model học tốt hơn bằng cách sử dụng các loss function truyền thống như contrastive loss [41], triplet loss[34], and center loss [45]. Do đó loss funciton cần được thiết kế 1 cách tinh tế.

    Lấy cảm hứng từ contrastive loss, range loss [73] được tạo ra nhằm giảm thiểu sự biến động trong cùng 1 lớp (intra-class variation) và đồng thời tăng cường khoảng cách giữa các lớp khác nhau (inter-class separation) trong hypershere.

        ![](images/4.1.3.%20Range%20loss%201.png)

        ![](images/4.1.3.%20Range%20loss%202.png)

    Zhong et al. [74] thì áp dụng **Noise Resistance (NR) Loss** - 1 based on large margin loss để train với các dữ liệu đầu tiên nhằm xử lý dữ liệu nhiễu và tối ưu hóa không gian đặc trưng trong các tập dữ liệu không cân bằng.
    
        ![](images/4.1.3.%20Noise%20Resistance.png)
    
    Sau đó sử dụng **Center-Dispersed Loss (CD Loss)**
    
        ![](images/4.1.3.%20CD%20loss.png)

2. **shallow data - dữ liệu nông**. Đây là tình huống mà dữ liệu huấn luyện bị hạn chế về số lượng ảnh cho mỗi ID (danh tính), khiến cho nhiều ID chỉ có một số ít mẫu.
    Trong rất nhiều các kịch bản FR trong thực tế, trainging dataset bị giới hạn về độ sâu. Chỉ 1 số nhỏ sample khả thi trong hầu hết các IDs.
    **Do đó, khi huấn luyện trên tập dữ liệu "nông" như vậy, mô hình sẽ có xu hướng thoái hóa (degeneration) và quá khớp (overfitting) do thiếu dữ liệu đa dạng cho từng lớp.**

    Ảnh hưởng của việc sử dụng Softmax Loss và các hàm mất mát có margin góc:
        Các hàm mất mát như Softmax Loss và những biến thể có margin góc (như CosFace) thường không đủ khả năng xử lý tốt các tập dữ liệu "nông".
        Trên các tập dữ liệu shallow, các hàm mất mát này dễ gây ra thoái hóa không gian đặc trưng (feature space collapse), khiến các vector đặc trưng không được phân biệt tốt giữa các ID, từ đó làm giảm hiệu quả nhận diện.
    Feature Space Collapse:
        Feature space collapse là hiện tượng khi các vector đặc trưng của các ID trong không gian đặc trưng trở nên gần nhau, khiến chúng không đủ khả năng phân biệt giữa các danh tính khác nhau.
        Khi dữ liệu không đủ phong phú, không gian đặc trưng bị thu hẹp lại (collapse), và mô hình sẽ không thể tạo ra các đặc trưng đa dạng cần thiết để phân biệt các ID khác nhau. Điều này dẫn đến việc mô hình không học được các đặc điểm phân biệt quan trọng giữa các danh tính.

    Li et al. [76] đề xuất khái niệm virtual class, giải pháp này dùng để xử lý các ảnh không có label. Thay vì yêu cầu dữ liệu có nhãn đầy đủ hoặc số lượng mẫu lớn cho mỗi ID, phương pháp này sử dụng dữ liệu không có nhãn và gán cho mỗi ảnh một lớp ảo (virtual class) để đại diện cho danh tính giả định của nó.
        Trong mỗi mini-batch, mỗi ảnh không có nhãn được xem là trung tâm của một lớp ảo (virtual class) và được coi như một lớp âm (negative class).
        Công thức mất mát (loss) được mở rộng bằng cách thêm các lớp ảo vào công thức của hàm mất mát dựa trên margin lớn (ví dụ như ArcFace):
            ![](images/4.1.3.%20Virtual%20class.png)

        Tăng cường đặc trưng: Để khai thác tốt hơn dữ liệu không có nhãn, một bộ sinh đặc trưng (feature generator) được thiết kế để tạo ra các đặc trưng tăng cường từ dữ liệu không có nhãn, từ đó giúp tăng tính phân biệt.

    ![](images/4.1.3.%20Meta-learning.png)

### Tổng kết

Bản chất 1 mô hình FR là thuộc về bài toán phân loại (classification). Đầu tiên model sẽ extract các feature của face image. Sau đó cố gắng tìm 1 space (hyperspace) nào đó để biểu diễn các feature này, sao cho các feature của cùng 1 face ID sẽ ở gần nhau và các feature của các image khác face ID sẽ xa, khác nhau.

Hyperspace

![](images/4.1.%20hyperspac.png)

![](images/4.1.%20hypershere.png)

Khoảng cách euclidean và góc.

![](images/4.1.%20Euclidean.png)

![](images/4.1.%20Góc.png)

![](images/4.1.%20Example.png)

Mỗi ảnh sau khi được model trích xuất sẽ cho ra 1 feature vector tương ứng với nhiều trường chứ không phải là 1 điểm (lý do là vì khuôn mặt phức tạp nên không tồn tại 1 điểm thỏa mãn). Các đặc trưng về hình dạng, độ sáng và các chi tiết khuôn mặt khác được giữ lại và thể hiện qua các giá trị trong vector đó. Do đó ta có thể dùng khoảng cách euclidean hoặc công thức cosin để tính toán khoảng cách/góc giữa 2 vector này. Trong không gian đặc trưng, việc so sánh giữa các vector trở nên hữu ích, vì chúng ta có thể dùng khoảng cách Euclidean hoặc góc cosine giữa các vector để đo độ giống nhau. Hai vector gần nhau hoặc có góc nhỏ giữa chúng sẽ đại diện cho hai ảnh có các đặc trưng tương tự.

Các phương pháp ở phần 4.1.1 cho ra hiệu xuất thấp hơn so với các phương  pháp ở phần 4.1.2. Và phương pháp mạnh nhất trong bài reseach này là MagFace và AdaFace (xem ở phần 6)

Ở mục 4.1.1, ta tìm hiểu các loss function có vai trò như 1 metric trong quá trình train. Mục tiêu của quá trình train là tối thiểu hóa giá trị loss này và bản thân Loss chính là khoảng cách euclidean hay cosin giữa các feature vector extract từ face image.
1. **cross-entropy**: Đối với lĩnh vực face verification, ta có thể sử dụng crosss-entropy làm loss function.
2. **contrastive loss**: *Kể từ loss này bắt đầu xuất hiện positive pair/ negative pair, margin m, khoảng cách euclidean/cosin*
    Mục tiêu của hàm loss này là sao cho khoảng cách giữa 2 feature thuộc cùng 1 face ID sẽ càng gần nhau và khoảng cách giữa 2 feature khác face ID sẽ cách xa nhau. Tức là hàm loss này sẽ thực hiện so sánh từng cặp image 1 trong dataset => Hội tụ rất lâu.
    Hàm này có thể được biểu diễn theo khoảng cách euclidean hoặc cosin

    ![](images/4.1.1.%20contrastive.png)
    
    ![](images/4.1.1.%20contrastive-cosin.png)

    **Trong đó margin m là tham số cố định phải truyền vào trong quá trình train.**
3. **BioMetricNet**
    Hiểu nôm la là nó rất phức tạp, mục tiêu của hàm này không liên quan gì đến khoảng cách mà là cố map các feature vector vào 1 space sao cho positive pair và negative pair sẽ thuộc vào 1 dạng phân phối nào đó => Không quan tâm vì không dùng
4. **Triplet loss**
    Đầu vào của nó gồm 3 ảnh anchor, positive (cùng ID với anchor) và negative (khác ID với anchor).
    Bối cảnh ra đời của hàm này là quá trình phân cụm các ảnh thuộc cùng 1 ID phải ở 1 chỗ riêng với chỗ các các face ID khác.
    **Mục tiêu của hàm này là tối đa khoảng cách (x^a,x^n) và tối thiểu hóa khoảng cách (x^a, x^p) và (x^a,x^n)-(x^a,x^p) > alpha (hay là margin)**. Trong đó x là feature của face được model extract từ image, x^a là anchor feature, x^p là postive feature, x^n là negative feature. Hiển nhiên quá trình này lặp lại với bộ 3 ảnh được lấy từ dataset => Hội tụ cực lâu.

    Để quá trình train diễn ra nhanh hơn => Hội tụ phải nhanh => Ta phải chọn ra các bộ 3 ảnh tệ nhất có thể xảy ra và làm cho nó thỏa mãn điều kiên train thì các bộ 3 khác chắc chắn thỏa mãn => Khái niệm hard positive và hard negative ra đời.
        hard-positive: khoảng cách (xp,xa) lớn nhất => Giúp mô hình học cách thu nhỏ khoảng cách giữa anchor và positive.
        hard-negative: khoảng cách (xn,xa) nhỏ nhất => Nhằm tăng độ khó cho mô hình khi cố gắng phân biệt anchor với negative.
5. **Kang loss**
    Kang và các cộng sự của anh đã đơn giản hóa contrastive loss và triplet loss vào 1 công thức.
    ...
6. **center loss**
    Phát triển thêm các hàm trên bằng cách giảm thiệu sự phân tán của các feature trong cùng 1 class, điều mà các loss function trên chưa làm được. Cụ thể, nó sẽ tìm 1 train space sao cho các feature thuộc cùng 1 face ID sẽ phân tán gần về phía trung tâm của class đó hơn.

    ![](images/4.1.1.%20center.png)

Ở mục 4.1.2, ta bắt đầu tìm hiểu 1 loại loss mới dựa trên margin. Margin là giá trị biên độ, đảm bảo khoảng cách giữa 2 cái gì đó luôn lớn hơn margin. Đây là dạng loss function có công thức chứ không phải là 1 metric learning như các loss ở trên.
1. Softmax Loss cơ bản
    Ý tưởng: Softmax loss là hàm mất mát cơ bản cho các bài toán phân loại, nhưng không tối ưu cho việc nhận diện khuôn mặt. Softmax dựa trên việc tính toán khoảng cách giữa feature vector của ảnh và trọng số của lớp để phân loại, nhưng không khuyến khích sự phân biệt rõ ràng giữa các lớp.

    Giới hạn: Thiếu khả năng tăng cường khoảng cách giữa các lớp khác nhau, dẫn đến không gian đặc trưng kém phân biệt (không có khả năng đẩy các feature vector của các class khác ra xa và làm feature của cùng class gần nhau).

2. L-Softmax và A-Softmax (SphereFace)
    Ý tưởng: Bổ sung margin góc vào softmax, thay đổi khoảng cách giữa feature vector của ảnh và trọng số lớp bằng cách thêm một giá trị margin vào góc giữa chúng.

    Cải tiến so với Softmax: Khuyến khích các vector của cùng một lớp nằm gần nhau hơn và cách xa các lớp khác. A-softmax đưa thêm margin vào góc giữa các đặc trưng của positive class và negative class, giúp các đặc trưng khuôn mặt dễ phân biệt hơn.

    Giới hạn: 
        Phức tạp trong huấn luyện, đặc biệt với dữ liệu nhiễu hoặc mất cân bằng.
        Huấn luyện phức tạp và yêu cầu điều chỉnh các siêu tham số nhạy cảm.

3. NormFace
    Ý tưởng: Chuẩn hóa embedding và trọng số lớp, sau đó mở rộng embedding lên một tham số 𝑠 để điều chỉnh sự phân bố các đặc trưng trên một hypersphere.

    Cải tiến: Tăng độ ổn định và khả năng phân biệt đặc trưng, giảm bớt sự phụ thuộc vào các margin phức tạp. Tăng cường tính ổn định, nhưng không sử dụng margin cho các lớp dương như SphereFace.

4. CosFace và ArcFace
    Ý tưởng: Thêm margin góc vào softmax để tạo khoảng cách giữa các lớp.
        CosFace trừ margin vào cosine của góc giữa đặc trưng và trọng số lớp.
        ArcFace đưa margin vào trong hàm cosine, tạo margin dạng cung (angular margin) để cải thiện tính phân biệt.
    Cải tiến: CosFace và ArcFace tăng độ chính xác nhận diện, tối ưu hóa phân tách giữa các lớp bằng cách trực tiếp thao tác trên góc.

    Giới hạn: Đòi hỏi điều chỉnh các siêu tham số nhạy cảm, dễ làm mất ổn định khi huấn luyện.

5. SV-Softmax
    Ý tưởng: SV-Softmax chọn các mẫu âm khó (hard negative) và đẩy nó ra xa trung tâm các lớp positive, giúp cải thiện sự phân tách nội lớp.

    Cải tiến: Đưa ra chiến lược hard negative mining, đẩy mạnh sự gắn kết của các mẫu trong lớp dương bằng cách xử lý các mẫu âm khó.

6. Ring Loss
    Ý tưởng: Giữ độ dài của các embedding không đổi ở giá trị 𝑅, giúp embedding có cùng độ lớn và tăng cường tính ổn định.

    Cải tiến: Hỗ trợ các hàm mất mát khác duy trì độ dài embedding, giảm thiểu ảnh hưởng của nhiễu trong không gian đặc trưng.

7. MagFace
    Ý tưởng: Điều chỉnh margin theo độ lớn của đặc trưng khuôn mặt theo chất lượng mẫu. Mẫu dễ sẽ có độ lớn cao và nằm gần trung tâm, mẫu khó hoặc nhiễu có độ lớn nhỏ và cách xa.
    
    Cải tiến: Kết hợp cả margin và độ lớn, giúp tối ưu hóa phân bố của các mẫu dễ và khó trong không gian đặc trưng, đồng thời tăng tính chống nhiễu.

8. AdaFace
    Ý tưởng: Điều chỉnh gradient của các mẫu khó dựa trên chất lượng của ảnh. Khi chất lượng cao, mẫu khó được nhấn mạnh; khi chất lượng thấp, mức độ ảnh hưởng của mẫu khó giảm.

    Cải tiến: Tự động điều chỉnh margin theo chất lượng dữ liệu, cải thiện tính ổn định của mô hình khi gặp dữ liệu đa dạng về chất lượng.

9. Sub-center ArcFace
    Ý tưởng: Chia các mẫu của một danh tính thành nhiều sub-center, với một sub-center chứa các mẫu sạch và các sub-center còn lại chứa mẫu khó hoặc nhiễu.

    Cải tiến: Giảm áp lực ràng buộc nội lớp, cải thiện khả năng phân loại khi có dữ liệu nhiễu bằng cách xử lý dữ liệu theo từng sub-class.

10. CurricularFace
    Ý tưởng: Áp dụng curriculum learning để học từ các mẫu dễ ở giai đoạn đầu và các mẫu khó ở giai đoạn sau của quá trình huấn luyện.

    Cải tiến: Cải thiện độ chính xác và tính phân biệt bằng cách tập trung vào mẫu dễ trước khi chuyển sang mẫu khó, cập nhật trọng số động qua Exponential Moving Average (EMA) để tối ưu hóa quá trình huấn luyện.

11. NPCface
    Ý tưởng: Nhấn mạnh các mẫu khó cả về dương và âm thông qua collaborative margin để xử lý các tập dữ liệu lớn, nơi các mẫu hard positive và hard negative thường xuất hiện cùng nhau.

    Cải tiến: Tăng khả năng phân biệt với các tập dữ liệu lớn, giúp mô hình tập trung vào các mẫu khó một cách toàn diện hơn.

12. UniformFace
    Ý tưởng: Phân bố đồng đều các lớp trong không gian đặc trưng trên một hypersphere, tối ưu hóa không gian đặc trưng bằng cách giữ khoảng cách tối thiểu giữa các lớp.
    
    Cải tiến: Tối đa hóa khả năng khai thác không gian đặc trưng, giảm thiểu hiện tượng chồng chéo giữa các lớp và tăng cường khả năng phân biệt.

## 4.2. Embedding
Khác với việc thiết kế các loss function ở trên, embedding refinement (tinh chỉnh embedding) là 1 cách khác để tăng cường kết quả FR.
- Ý tưởng 1: thiết lập các ràng buộc rõ ràng giữa face embeddings với face generators (trình tạo khuôn mặt), giúp cải thiện khả năng phân biệt của các vector đặc trưng trong không gian đặc trưng.
- Ý tưởng 2: thay đổi face embeddings bằng các thông tin phụ trợ lấy được từ training image như độ che khuất (occlusion), độ phân giải hỉnh ảnh (resolution) nhằm cải thiện độ chính xác nhận diện ngay cả khi có các yếu tố nhiễu.
- Ý tưởng 3: models FR in multi-tasks. Các tasks như đoán tuổi và tư thế cũng đã được thêm vào mạng, giúp mô hình học được các đặc trưng bổ sung có lợi cho việc phân biệt khuôn mặt.

### 4.2.1. Embedding refinement by face generator (Tinh chỉnh Face Embedding bằng face generator)

1. **DR-GAN[78] - Deep Reconstruction Generative Adversarial Network**
    **Các giải pháp FR dựa trên face generator thường dựa trên yếu tố face sẽ bất biến theo tuổi hoặc tư thế (góc chụp). => Invariance (tính bất biến)**
    DR-GAN[78]: đã giải quết vấn đề face bất biến theo tư thế bằng cách tổng hợp các khuôn mặt có tư thế khác nhau. Cụ thể **mạng này sẽ học cách biểu diễn face image bởi 1 kiến trúc encoder-decoder generator**. Đầu ra của decoder tổng hợ nhiều khuôn  mặt của cùng 1 ID với các pose (tư thế) khác nhau.
        encoder (Genc): đưa vào ID label y^d và label tư thế y^p của face image x. Encoder sẽ extract tư thế trong ảnh tạo ra emdedding dựa theo tư thế gốc của ảnh f(x) = Genc(x). Sau đó f(x) được concate với pose code c (mã tư thế) và 1 chỉ số random noise z.
        decoder (Gdec): tạo ra các face image với các tư thế khác nhau bằng cách giải mã embedding Gdec = (f(x), c, z) và hiển nhiên label vẫn thế.

    Với mỗi ảnh được tổng hợp (synthetic) được tạo ra bởi phương pháp trên (bộ generator), disciminator D (bộ phân biệt D) sẽ cố gắng ước tính xem ảnh mới x~ cùng tư thế của nó có phải là ảnh fake hay không.

    **Tóm lại là học đối kháng**

2. **D2AE - Identity Distilling and Dispelling Auto-encoder** (chưng cất và phân tán)
    **Liu et al. [79] đề xuất sử dụng một mạng auto-encoder** để học các đặc trưng khuôn mặt phục vụ cho xác minh danh tính và nhằm cải thiện độ chính xác bằng cách tạo ra các đặc trưng tách biệt rõ ràng cho danh tính (identity-distilled) và các yếu tố khác ngoài danh tính (identity-dispelled).

    Cụ thể encoder nhận vào 1 ảnh x và cố gắng extract feature của ảnh. Sau đó nó cố gắng tách feature này thanh 2 nhánh. 
        Distilling Bt: Nhánh chỉ chứa các feature phục vụ xác minh danh tính và được tối ưu hóa bởi hàm softmax
        Dispelling Bp: Nhánh cố gắng loại bỏ các feature liên quan đến danh tính trong feature gốc của ảnh. Kết quả là 1 feauture chứa các yếu tố như ánh sáng, góc chụp, ... . Nhánh này cố gắng đánh lừa decoder bằng cách tạo ra 1 phân phối danh tính cố định.

    Decoder nhận cả 2 vector fT và fP ở trên làm đầu vào và đảm bảo 2 cái này được bảo toàn trong không gian đặc trưng

    ![](images/4.2.1.%20D2AE%20-%20arichitech.png)

    ![](images/4.2.1.%20D2AE.png)

3. **R3AN**
    Chen et al. [80] đề xuất để giải quyết bài toán Cross Model Face Recognition (CMFR) bằng mạng R3AN.
    
    CMFR là bài toán nhận diện khuôn mặt bằng cách nhận dạng đặc trưng đầu vào từ một mô hình bằng cách sử dụng dữ liệu từ một mô hình khác. 
    
    Phương pháp R3AN gồm ba thành phần chính: reconstruction (tái tạo), representation (đại diện đặc trưng), và regression (hồi quy).

    ![](images/4.2.1.%20R3AN.png)

4. **MTLFace**
    Huang et al. [81] đề xuất, một phương pháp học sâu kết hợp để xử lý đồng thời bài toán nhận diện khuôn mặt không phụ thuộc tuổi (age-invariant face recognition, AIFR) và dự đoán tuổi của khuôn mặt (face age synthesis, FAS)

    MTLFace sử dụng một kiến trúc encoder-decoder, cùng với cơ chế attention và adversarial learning để tạo ra embedding không bị ảnh hưởng bởi tuổi tác và có khả năng tổng hợp khuôn mặt ở các độ tuổi khác nhau.

    ![](images/4.2.1.%20MTLFace.png)

5. **TS-GAN -Teacher-Student Generative Adversarial Network ***
    Tạo ra hình ảnh độ sâu từ hình ảnh RGB đơn (màu) nhằm cải thiện hiệu suất của hệ thống nhận diện khuôn mặt.
    Cấu trúc: Bao gồm hai thành phần chính là giáo viên (teacher) và học sinh (student).
        Giáo viên: Gồm một generator (tạo hình) và một discriminator (phân loại) học cách ánh xạ giữa kênh RGB và độ sâu từ hình ảnh RGB-D (hình ảnh có độ sâu).
        Học sinh: Cải thiện ánh xạ đã học cho hình ảnh RGB bằng cách huấn luyện thêm generator.
    Quá trình huấn luyện: Mô hình nhận vào hình ảnh RGB và tạo ra hình ảnh độ sâu, sau đó trích xuất các đặc trưng của cả hai hình ảnh một cách độc lập. Cuối cùng, các đặc trưng của RGB và độ sâu được kết hợp để dự đoán ID khuôn mặt.

6. Liên quan đến ảnh không gán nhãn => Bỏ qua.

### 4.2.2. Embedding refinement by extra representations (tinh chỉnh embedding bằng các tham số phụ khác)

2 giải pháp [85] và [86] ở trên đều coi face embedding như low-rank (tức embedding có ít chiều). Họ chỉ thêm nhiễu vào các image giúp cải thiện khả năng học của mô hình (giúp mô hình có thể bền vững trước những tác động nhỏ bên ngoài). Việc này có thể được chia thành hai phần: tái cấu trúc các đặc trưng khuôn mặt một cách tuyến tính từ một từ điển (dictionary) và áp dụng các ràng buộc thưa (sparsity constraints).

1. **Neural Aggregation Network - NAN**
    **Phương pháp FR bằng video điển hình** bằng cách thao túng các face embedding.

    Yang et al. đề xuất rằng, các hình ảnh khuôn mặtcủa cùng 1 IDs trong một video với cùng một ID nên được gộp lại để xây dựng một embedding mạnh mẽ hơn (robots embedding).

    ![](images/4.2.2.%20NAN.png)

2. **Dynamic Feature Matching**
    **Phương pháp này nhằm giải quyết vấn đề nhận diện khuôn mặt khi chỉ có một phần của khuôn mặt được hiển thị**, do che khuất hoặc do góc nhìn không thuận lợi.

    Đầu tiên, một mạng nơ-ron tích chập được áp dụng để trích xuất các đặc trưng từ hình ảnh khuôn mặt probe (hình ảnh cần nhận diện) và hình ảnh trong bộ sưu tập (gallery) có kích thước tùy ý. Hình ảnh probe được ký hiệu là 𝑝 và hình ảnh gallery được ký hiệu là 𝑔𝑐 (với 𝑐 là nhãn của hình ảnh gallery).

    Thông thường, việc tính toán độ tương đồng giữa 𝑝 và 𝑔𝑐 gặp khó khăn do kích thước cả 2 feature không nhất quán. Để khắc phục điều này, một cửa sổ trượt (sliding window) có kích thước giống như 𝑝 được sử dụng để phân tách 𝑔𝑐​ thành 𝑘 feature con: 𝐺𝑐=[𝑔𝑐1,𝑔𝑐2,…,𝑔𝑐𝑘].

3. Pose Invariant Model (PIM)
    Một phương pháp được Zhao et al. đề xuất nhằm cải thiện việc nhận diện khuôn mặt (Face Recognition - FR) trong môi trường tự nhiên, khi khuôn mặt có thể bị nghiêng hoặc không đối diện trực tiếp với máy ảnh.

    PIM được thiết kế để làm cho việc nhận diện khuôn mặt trở nên bất biến với góc nhìn (pose invariant), giúp nhận diện khuôn mặt hiệu quả ngay cả khi góc nhìn của khuôn mặt không phải là chính diện.

    Mạng con Face Frontalization Sub-Net (FFSN): để chuyển đổi các hình ảnh khuôn mặt nghiêng thành hình ảnh khuôn mặt chính diện.
        Hình ảnh khuôn mặt nghiêng ban đầu được đưa vào một bộ phát hiện các điểm đặc trưng trên khuôn mặt để xác định các mảng đặc trưng (landmark patches).
        Các mảng đặc trưng từ hình ảnh khuôn mặt nghiêng này, gọi chung là 𝐼𝑡𝑟​ , được đưa vào PIM.
        Sau đó, PIM sử dụng một cấu trúc mã hóa-giải mã (encoder-decoder) được ký hiệu là 𝐺 để tạo ra hình ảnh khuôn mặt chính diện từ 𝐼𝑡𝑟​, ký hiệu là 𝐼′=𝐺(𝐼𝑡𝑟).
        Tương tự như GAN, một mạng học phân biệt (discriminative learning sub-net) được kết nối với FFSN nhằm đảm bảo rằng hình ảnh khuôn mặt chính diện tạo ra, 𝐼′, trông giống như một khuôn mặt thực và mang thông tin về danh tính.

        Kết quả: Các đặc trưng từ khuôn mặt nghiêng và khuôn mặt chính diện được tạo ra sẽ được kết hợp để có được một biểu diễn khuôn mặt (face representation) tốt hơn, giúp tăng độ chính xác khi nhận diện khuôn mặt.

### 4.2.3. Multi-task modeling with FR

Bên cạnh Face Ids, rất nhiều các phương pháp chọn đưa thêm nhiều supervised information trong quá trình train mô hình FR.

1. **Peng et al. [103]** giới thiệu 1 phương pháp học biểu diễn các feature sao cho nó không phụ thuộc vào tư thế của khuôn mặt **pose-invariant**.
    Đầu tiên 3D facial model sẽ được sử dụng để tạo ra ảnh có góc nhìn mới so với góc nhìn gần hướng chính diện ban đầu.

    Ngoài ID của ảnh, còn có các thông tin khác như face pose (tư thế khuôn mặt), landmark (đặc điểm khuôn mặt như tai mũi họng, ...) cũng được đưa vào giám sát trong quá trình train. Giúp mô hình học được các đặc trưng phong phú hơn bằng cách học đồng thời cả đặc trưng nhận dạng và đặc trưng không liên quan đến nhận dạng với bộ trích xuất đặc trưng (extractor θᵣ).
        ei: ID labels
        ep: pose labels
        ei: landmark labels
    
    Rich embedding (1 thuật ngữ để chỉ rằng embedding chứa phong phú các thông tin) trong quá trình train sẽ được chia làm các feature về identity, pose và landmark feature. Và để train ra các feature này, ta sẽ sử dụng các loss khác nhau. 
        softmax cho việc ước tình ID
        L2 regression được sử dụng cho pose và landmark preidict.
    
    Cuối cùng 1 cặp ảnh: ảnh có tư thế gần chính diện x1 và 1 ảnh có tư thế không phải chính diện x2 được đưa vào mạng recognition θr để trích xuất embedding er1 và er2.

    Họ sử dụng kỹ thuật disentangling (tách biệt) dựa trên reconstruction để  chắt lọc những feature liên quan đến identity feature khỏi những feature không liên quan đến identity.
    
    Kết qủa sau cùng ta thu được 1 embedding được biểu diễn không phụ thuộc vào pose của khuôn mặt, giúp cải thiện khả năng nhận dạng trong các tính huống mặt có pose khác nhau.

    ![](images/4.2.3.%20chịu.png)

2. **Wang et al. [104]** giải quyết vấn đề hệ thống nhận diện không phụ thuộc vào tuổi bằng cách thêm 1 task predict age (dữ liệu dùng để train là ảnh của người đó tại 1 độ tuổi không đổi trong khi các đặc trưng của khuôn mặt có thể thay đổi theo thời gian). 
    Nhóm tác giả đề xuất Orthogonal Embedding CNNs (OE-CNNs) nhằm học các đặc trưng khuôn mặt không phụ thuộc vào tuổi tác.
    Đầu tiên họ train 1 mô hình face feature extractor để thu được feature xi của sample i. Sau đó xi sẽ được tách làm 2 thành phần:
        Thành phần liên quan đến danh tính xid: Được tối ưu bởi SphereFace[47]
        Thành phần liên quan đến độ tuổi xage: Được tối ưu bỏi hàm loss sau.
            ![](images/4.2.3.%20Loss%20age%20OE%20CNN.png)

            Trong đó ||xi||2 là độ dài của embedding xi, zi là label về tuổi. M là batch size.

3. **Liu et al. [105]** hợp nhất 3D face reconstruction (tái tạo khuôn mặt 3D) và recognition (nhận dạng khuôn mặt).
    Sử dụng dữ liệu point cloud.
        ![](images/4.2.3.%20point%20cloud.png)
    Nhóm tác giả cho rằng 3D shape có thể được chia thành các thành phần liên quan đến identity hoặc không liên quan đến identity.
        ![](images/4.2.3.%20các%20thành%20phần%20của%20mặt%203d.png)
    Sau đó 1 encoder được built để extract face feature từ 1 ảnh 2D. Feature này được chia thành 2 thành phần cid (các feature liên quan đến định danh, phục vụ cho việc nhân dạng) và cres (các feature mô tả hình dạng khuôn mặt, không liên quan đến danh tính):
        ![](images/4.2.3.%20Liu%20encoder.png)
    Các hàm loss LC, LR được thiết kế cho việc predict identity và recontruction 3D shape.
        LC là softmax để tối ưu cid
    Tổng quan kiến trúc mạng multi task 
        ![](images/4.2.3.%20Tái%20tạo%203D.png)

4. **Wang et al. [106]** đề xuất mạng FM2u-Net được thiết kế để tạo ra khuôn mặt có trang điểm, nhằm hỗ trợ các tác vụ xác thực khuôn mặt không phụ thuộc vào trang điểm.
    Kiến trúc FM2u-Net
        **FM-Net (Face Morphological Network): Dùng để tạo ra khuôn mặt có trang điểm.**
            Cycle consistent loss: Được sử dụng để hướng dẫn quá trình huấn luyện tạo ảnh trang điểm giống thật.
            Dữ liệu huấn luyện: Do thiếu các cặp dữ liệu có và không có trang điểm, FM-Net sử dụng ảnh gốc và các miếng vá khuôn mặt (facial patches) làm thông tin giám sát.
            Hạn chế thay đổi danh tính:
                Thêm softmax loss để dự đoán ID.
                Thêm ID-preserving loss để đảm bảo khuôn mặt tạo ra vẫn giữ được danh tính ban đầu.
        **AttM-Net (Attention Makeup Network): Dùng để học đặc trưng khuôn mặt không bị ảnh hưởng bởi trang điểm.**
            gồm:
                Một nhánh toàn cục (global branch).
                Ba nhánh cục bộ (local branches): Tập trung vào hai mắt và miệng.
            Mục đích: Học đặc trưng tổng quát và chi tiết của các phần khuôn mặt, từ đó tạo ra đặc trưng không bị ảnh hưởng bởi trang điểm.
            
            Kết hợp đặc trưng (Feature Fusion): Kết hợp đặc trưng của các phần khác nhau lại thành một đặc trưng tổng hợp fcls
            
            Hàm loss:​               
                ![](images/4.2.3.%20attm%20loss.png)

5. **Gong et al. [108]** đề xuất 1 mạng đối nghịch khử thiên vị (DebFace) để cùng học FR và các thuộc tính nhân khẩu học như giới tính, tuổi và chủng tộc.
    DepFace network gồm 4 thành phần.
        Encoder để trích xuất feature từ image:
        Các bộ classifer bao gồm: CG để phân loại giới tính, CA để phân loại tuổi, CR để phân loại chủng tộc và CID để phân loại danh tính
        Bộ phân loại phân phối CDistr
        Mạng tổng hợp các feature sau cùng EF eat
    
## 4.3. FR with massive IDs
Càng nhiều dataset sẽ giúp hệ thống FR tốt hơn, tuy nhiên đi kèm với nó là những vấn đề, thách thức mới. Chi phí tính toán và bộ nhớ: Số lượng ID lớn trong tập huấn luyện có thể cải thiện kết quả FR, nhưng đồng thời cũng làm tăng chi phí tính toán và bộ nhớ. Việc mở rộng số lượng lớp phân loại có thể vượt quá khả năng của GPU, dẫn đến việc cần các giải pháp để tối ưu hóa.

1.  Partial FC [112][113]
    Giải pháp đầu tiên được đề xuất là phân tách lớp phân loại theo chiều lớp và phân phối đều trên các GPU. Partial FC đề xuất rằng không cần sử dụng tất cả các trung tâm lớp âm khi tính toán logits; chỉ cần lấy mẫu một phần trung tâm lớp âm cũng đủ để đạt được độ chính xác tương đối cao.

2. BroadFace: Đề xuất của BroadFace là giữ lại các đặc trưng của các vòng lặp trước đó trong một hàng đợi và tối ưu hóa các tham số của lớp phân loại cùng với các đặc trưng hiện tại, nhằm nâng cao số lượng mẫu tham gia vào mỗi lần tối ưu hóa.

3. Vấn đề về không gian đặc trưng: Khi tiến trình huấn luyện diễn ra, không gian đặc trưng của mô hình có thể trôi dạt, tạo ra khoảng cách giữa các đặc trưng trong hàng đợi và không gian đặc trưng hiện tại. Để khắc phục, các tham số của lớp phân loại ở vòng lặp trước đó được sử dụng để điều chỉnh các đặc trưng trong hàng đợi.

4. Đề xuất mới (Virtual FC layer): Một lớp mới gọi là Virtual FC layer được đề xuất để giảm thiểu mức tiêu thụ tính toán. Nó chia số lượng ID trong tập huấn luyện thành các nhóm và sử dụng một ma trận chiếu để chia sẻ các cột giữa các nhóm.

5. Faster Face Classification (F2C): Một phương pháp mới được đề xuất để lưu trữ và cập nhật các đặc trưng của danh tính một cách động, có thể được xem là một sự thay thế cho lớp phân loại thông thường

## 4.4. Cross domain in FR
Nhìn chung khi sử dụng các thuật toán FR, training set và testing set thường trong 1 bối cảnh giống nhau (similar distribution). Tuy nhiên face images từ các chủng tốc khác nhau, trong các bối cảnh khác nhau (mobile photo albums, online video, Căn cước công dân, ...) đều có độ lệch miền rõ ràng (domain bias). Hiệu suất model sẽ giảm mạnh khi training set và testing set có khoảng cách miền. Điều này dẫn đến khả năng tổng quát hóa kém của mạng neutron trong việc xử lý dữ liệu chưa từng thấy trong thực tế. Do đó các giải pháp thích ứng miền đã được đề xuất để giải quyết vấn đề này.

Phần này nói về các giải pháp thích ứng miền chung (general domain adaptation), sau đó liệt kê 1 số phương pháp áp dụng cho các tập training set đặc biệt

1. MAML (Model-Agnostic Meta-Learning): được rất nhiều researcher sử dụng để giải quyết cross domain.
    Guo et al. [119] đã đề xuất 1 phương pháp nhận diện khuôn mặt meta (MFR - Meta Face Recognition) để xử lý vấn đề này.

    Trong mỗi vòng lặp huấn luyện, chỉ một trong số N miền trong tập huấn luyện được chọn làm dữ liệu thử nghiệm meta (meta-test data), trong khi N-1 miền còn lại được sử dụng làm dữ liệu huấn luyện meta (meta-train data). Tất cả các dữ liệu này tạo thành một meta-batch. Dữ liệu thử nghiệm meta được sử dụng để mô phỏng hiện tượng chuyển miền trong các kịch bản ứng dụng.

    Ba loại hàm mất mát được đề xuất bao gồm:

        Hàm mất mát attention hard-pair (Lhp): Tối ưu hóa các cặp tích cực và tiêu cực bằng cách giảm khoảng cách Euclidean giữa các cặp tích cực khó và đẩy các cặp tiêu cực khó ra xa.
        Hàm mất mát phân loại mềm (Lcls): Dùng cho phân loại ID khuôn mặt, được điều chỉnh từ hàm mất mát cross-entropy.
        Hàm mất mát định hướng miền (Lda): Thiết kế để làm cho các vector nhúng trung bình của nhiều miền huấn luyện meta gần nhau hơn.

2. Faraki và các cộng sự nhận thấy vấn đề trong sử dụng hàm mất mát định hướng miền (domain aligment loss, Lda) trong phương pháp MFR.
    Họ nhận thấy rằng việc kéo gần giá trị trung bình của các miền có thể dẫn đến giảm hiệu suất của mô hình, vì khi đó các mẫu thuộc các ID khác nhau có thể bị kéo gần lại với nhau, làm giảm độ chính xác.

    Để khắc phục vấn đề này, Faraki et al. đã đề xuất một hàm mất mát mới gọi là cross domain triplet loss (mất mát ba phần trong các miền khác nhau), dựa trên hàm mất mát triplet. Công thức cho hàm mất mát này được mô tả trong đoạn văn và bao gồm các thành phần như sau:
        Các triplet: Gồm ba phần: anchor (mẫu gốc), positive (mẫu tích cực) và negative (mẫu tiêu cực) trong không gian đặc trưng.
        Khoảng cách Mahalanobis: Được tính toán dựa trên các cặp tích cực trong miền, nhằm điều chỉnh khoảng cách giữa các mẫu trong các miền khác nhau.   
    
    Hàm mất mát ba phần trong các miền khác nhau giúp căn chỉnh phân phối giữa các miền khác nhau, nhằm cải thiện độ chính xác của mô hình.

... tự đọc chịu chắc không dùng ...

## 4.5. FR pipeline acceleration (tăng tốc độ xử lý pipeline của hệ thống FR)

1. Network Slimming:
    Được giới thiệu bởi Liu et al. Ý tưởng chính của phương pháp xoay quay scaling factor trong mối bước batch normalization (BN) có thể chỉ ra mức độ quan trọng của mỗi channel. Tác giả sử dụng L1 regularization dựa trên scaling factor trong BN layer trong quá trình train để làm nó thưa thớt, sau đó cắt bỏ các kênh không quan trọng theo size của scaling factor.
        scaling factor được sử dụng để điều chỉnh độ lớn đầu ra của mỗi kênh (channel) trong mạng neutron. Cụ thể, sau khi đầu vào của một kênh được chuẩn hóa trong lớp BN, nó sẽ được nhân với hệ số tỉ lệ này để điều chỉnh giá trị đầu ra.
        Ví dụ 1 mạng CNN tiếp nối bởi 1 lớp BN.
            Convolutional Layer: 10 kênh (filters), kích thước kernel là 3×3.
            Batch Normalization Layer: đi sau lớp convolution để chuẩn hóa đầu ra của các kênh.
            Fully Connected Layer (hoặc có thể nhiều lớp convolution khác).
    Kết quả thực nghiệm cho thấy, structure của mạng quan trọng hơn weight và việc cắt tỉa được coi là quá trình tìm kiếm structure mạng thích hợp. Sau khi cắt tỉa, việc train mạng với 1 chuỗi các tham số khởi tạo bất kỳ sẽ cho mạng đạt kết quả tốt hơn.

2. K-D tree
    Trong vấn đề face identication, chúng ta cần khớp đặc điểm face feature của input với 1 face feature trong gallery. Đây là vấn đề 1:n và đề đẩy nhanh quá trình này K-D tree là 1 thuật toán phổ biến được sử dụng.

    K-D tree liên tục chia feature space thành 2 phần để tạo thành cấu trúc cây nhị phân. Mỗi point trên feature space tương ứng với 1 node trong tree. Sau đó chỉ cần chạy thuật toán tìm kiếm gần nhất

    Hạn chế của phương pháp: khi feature space có nhiều chiều, dữ liệu thường phân bổ rất thưa thớt khiến cho K-D suy giảm hiệu quả (gần như trở thành tìm kiếm tuyến tính). Việc phân đoạn không gian đặc trưng kém hiệu quả, dẫn đến độ chính xác của tìm kiếm cũng giảm xuống.

3. Vector Quantization và Product Quantization
    Các thuật toán này mã hóa không gian đặc trưng thành một số lượng điểm giới hạn (codebook), giúp giảm số lần tính toán trong việc so khớp đặc trưng. Product quantization tiếp tục chia không gian đặc trưng thành các sub-vectors và áp dụng codebooks cho từng nhóm sub-vectors.

4. Thuật toán tìm kiếm dựa trên đồ thị
    Trong đó, các điểm trong không gian đặc trưng được kết nối thành đồ thị. NSW (Navigable Small World) là một thuật toán tiêu biểu trong nhóm này, giúp cải thiện việc tìm kiếm nhanh chóng nhờ vào liên kết ngắn và dài, tạo ra các "đường tắt" trong đồ thị.

5. Locality Sensitive Hashing (LSH): 
    Thuật toán này sử dụng hàm băm để phân loại các vector gần nhau thành cùng một giá trị băm, giúp giảm số lượng vector cần so sánh trực tiếp, từ đó tăng hiệu quả tìm kiếm.

6. Knowledge Distillation
    Đây là phương pháp nén mô hình lớn (CNN) thành các mô hình nhẹ hơn để phù hợp với thiết bị di động. Phương pháp này bao gồm việc tối ưu hóa và giảm thiểu độ tương tự giữa các đặc trưng, đảm bảo mô hình nhỏ vẫn duy trì được hiệu năng từ mô hình gốc.

## 4.6. Closed-set Training

FR thường là vấn đề clossed-set training. Closed-set training nghĩa là các ảnh được sử dụng test với modle sau khi train sẽ có IDs nằm trong training sets. Tóm lại nhờ vào việc các IDs trong tập testset có nằm trong training set hay không ta có thể phân loại hệ thống FR thành closed-set systems và open-set systems.

Tong et al. [142] đã đề xuất 1 framework để đánh giá độ hệ thống FR với clossed-set và open-set. Kết quả thử nghiệm cho thấy rằng, hệ thống FR tập mở dễ bị tổn thương hơn hệ thống tập đóng dưới các loại tấn công khác nhau (tấn công kỹ thuật số, tấn công vật lý thực hiện ở cấp độ pixel và tấn công thục hiện ở cấp độ grid). Tóm lại closs-set training dễ hơn với open-set training.

## 4.7. Mask face recognition
FR đã đạt được những tiến bộ đáng kể trong vài năm gần đây. Tuy nhiên, khi áp dụng mô hình FR đó vào các tình huống không bị hạn chế, hiệu suất nhận dạng khuôn mặt giảm mạnh, đặc biệt là khi khuôn mặt bị che khuất bởi khẩu trang. Các phương pháp FR hiện đại giải quyết bài toán này thường là biến thể của 2 cách tiếp cận.
- Khôi phục các phần khuôn mặt bị che khuất - recovering occluđe facial parts [143, 144, 145, 146]
- Loại bỏ các đặc điểm bị hỏng do che khuất [147, 148, 149]: cách này thường được ưu tiên hơn vì không có gì đảm bảo việc khôi phục diễn ra thuận lợi.

1. Mask Decoder và Occlusion Pattern: được FROM[150] đề xuất để dự đoán phần sample bị che khuất.

    ![](images/4.7.%20FROM.png)

    Kiến trúc như trên hình

    Đầu tiên nó sẽ lấy mini-batch images làm đầu vào. Thông qua Feature Pyramid Extractor sẽ thu được 3 scale feature maps như X1, X2, X3. Trong đó X3 được sử dụng để giải mã (decode) the mask - cái mà chứa thông tin vị trí bị che khuất.

    Sau khi có được mask, nó sẽ được apply vào X1 để che khuất các điểm bị bỏng và kất feature thu được sau cùng làm đầu vào cho hệ thống nhận dạng.

    Cuối cùng  Occlusion Pattern Predictor dự đoán các mẫu occlusion như 1 sự giám sát bổ xung.

    Mô hình tạo ra các đặc trưng ở ba cấp độ khác nhau, sau đó dự đoán vị trí các vùng bị che khuất để loại bỏ phần đặc trưng bị nhiễu, giữ lại phần đặc trưng sạch cho nhận diện. Loss tổng thể là sự kết hợp của loss nhận diện khuôn mặt (CosFace loss) và loss dự đoán mẫu che khuất (MSE hoặc Cross-entropy loss).

2. Mô hình MSML (Multi-Scale Mask Learning):
    Mô hình MSML sử dụng học đa phân đoạn để xử lý các đặc điểm che khuất khác nhau.
    
    Cấu trúc gồm nhánh nhận diện khuôn mặt (Face Recognition Branch - FRB), nhánh phân đoạn che khuất (Occlusion Segmentation Branch - OSB), và các phép mask đặc trưng.
    
    Loss tổng thể gồm loss phân đoạn và loss nhận diện. 

3. Phương pháp với Mô hình CR (Channel Refinement):
    Phân chia các đặc trưng để dự đoán che khuất và nhúng danh tính (identity embedding).

    CR Network chuyển mặt nạ 2D thành 3D để phù hợp hơn với các bản đồ đặc trưng.
    
    Loss gồm CosFace loss để tối ưu nhận diện và các loss khác cho dự đoán mặt nạ.

4. Consistent Sub-decision Network: 
    Sử dụng các quyết định phụ (sub-decisions) dựa trên các vùng khuôn mặt khác nhau, áp dụng KL divergence để hướng mạng tập trung vào các phần khuôn mặt không bị che khuất.

    Áp dụng kiến thức truyền đạt (knowledge distillation) để đưa đặc trưng khuôn mặt bị che hướng tới đặc trưng khuôn mặt bình thường, giảm thiểu mất mát thông tin.

5. Phương pháp chiến thắng của thử thách ICCV 2021-MFR:
    
    Sử dụng kỹ thuật ánh xạ mặt nạ vào kết cấu khuôn mặt và sinh ảnh khuôn mặt bị che.

    Xây dựng khung làm sạch dữ liệu dựa trên học tự động, sử dụng DBSCAN để làm sạch dữ liệu nhận diện.

    Đề xuất Balanced Curricular Loss để điều chỉnh tầm quan trọng của các mẫu dễ và khó trong các giai đoạn huấn luyện khác nhau.

## 4.8. Privacy-Preserving FR (bảo vệ quyền riêng tư)
BỎ