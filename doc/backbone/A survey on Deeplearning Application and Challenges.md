# 1. Introduction
- Bài báo đề cập đến ứng dụng của STOA và những thách thức công nghệ này gặp phải trong từng lĩnh vực, giải pháp nào được ưu tiên sử dụng trong lĩnh vực nào.\
- Bài báo này còn đề cập đến nguyên tắc cơ bản hình thành nên các kiến trúc deeplarning nói chung.
- Mục lục
    - Phần 2: Các nguyên tắc cơ bản trong deep learning như layers, attention mechanisms, activation functions, model optimization, loss function, regularization, ...
    - Phần 3: Các Deep learning models và các kiến trúc mạng CNN
    - Phần 4: Ứng dụng thực tiễn của học sâu (quan trọng nhất)
    - Phần 5: Thảo luận về những thách thức, định hướng trong tương lai trong lĩnh vực deep learning.
    - Phần 6: Kết luận

# 5. Challenges and Future Directions
## 5.1. Avaiability and Quaylity (tính khả dụng và chất lượng)
Dataset phải có nhiều sample và càng đa dạng càng tốt thì mới cho ra chất lượng và tính tổng quát cao.
    - Độ phức tạp của kiến trúc mạng neutron (complexity) ảnh hưởng đến vấn đề over fitting (thường do kiến trúc mạng quá phức tạp và mô hình ít dữ liệu)
    - Độ tin cậy của dataset, liệu có sample sai lệch nào được đưa vào trong dataset không, điều này có thể ảnh hưởng đáng kể đến hiệu suất của model => 1 trong những cách tiếp cận giải quyết vấn đề này là transfer learning (sử dụng 1 mô hình đã được train trên 1 tập dữ liệu lớn để đào tạo tinh chỉnh lại với tập dữ liệu nhỏ hơn và cách này giảm thiểu tình trạng thiếu dữ liệu trong target domain).
    - Cách tiếp cận khác để giải quyết vấn đề này là data augmentation. Các thao tác như rotation, cropping, translation (tịnh tiến), ... Chú ý cần xem xét output nó có phản ánh thực tế không, ví dụ như các hiệu ứng như noise, flip có thể thay đổi cấu trúc các điểm, ... Nói chung cần cẩn thận khi chọn chiến lược data augmentation cho phù hợp.

## 5.2 Interpretability and Explainability (Khả năng diễn giải và khả năng giải thích)

Khả năng này rất quan trọng để xây dựng lòng tin và hiểu cách model predict, đặc biệt trong các lĩnh vực rủi do cao như y tế, chăm sóc sức khỏe. Tuy nhiên khi mạng neutron càng ngày càng deep, ta thường coi nó là 1 hộp đen và khó giải thích được chúng. Researchers sẽ phải tập trung vào các phương pháp cung cấp thông tin, làm thế nào để model đưa ra quyết định từ feature map ban đầu để model trở nên minh bạch đáng tin cậy hơn. 1 số phương pháp explain có thể kể đến như
- visualization: Làm nổi bật
- model distillation
- intrinsic (tự giải thích): Multi-task để có thêm thông tin giải thích cho chính quyết định của nó

## 5.3. Ethics and Fairness (Đạo đức và công bằng)

Deep learning ngày càng được đưa vào để áp dụng trong những lĩnh vực có độ rủi do cao như tuyển dụng, tư pháp, ... Tuy nhiên không có gì đảm bảo mô hình học chính xác 100%, những trường hợp sai lệch có thể dẫn đến vấn đề mất công bằng.

Nhiều phương pháp đã được thực hiện nhằm giảm thiểu vấn đề này. ....

## 5.5 Adversarial Attack and Defense (tấn công và phòng thủ)

...

## 6. Conclusion

# 2. Fundamentals of Deep Learning

## 2.1. Layers
- Input layer, hideen layer (nhiều lớp chịu trách nhiệm extract và xử lý các thông tin phức tạp feature map) và output layer
- Fully connected layers
- Cấu trúc của 1 neutron thường và neutron CNN (CNN chủ yếu được sử dụng để xử lý các dữ liệu có cấu trúc như image, time series).
- Pooling layers thường được áp dụng sau CNN layers để giảm dần kích thước của không gian (cao và rộng) của feature map.

## 2.2 Attention mechanisms

Không phải phần nào của input đều đáng quan trọng phải học. Ví dụ như backgroud của ảnh không có ý nghĩa trong quá trình recognition. Trong conv, tất cả các feauture đều được xử lý thông nhất mà không xem xét mức độ quan trọng khác nhau của các thành phần dữ liệu khác nhau của input. 

Hạn chế này được giải quyết bởi cơ chế attention mechanisms - cho phép mô hình gán trọng số cho các features từ đó biết thằng nào quan trọng hơn. Nhờ tính năng này, model ưu tiên học các khía cạnh quan trọng hơn của vật thể, tăng độ chính xác khi ra quyết định.
    A = f(g(x), x)

0. Channel attention
    - Squeeze-and-Excitation (SE) Attention:
        Sử dụng global average pooling (GAP) để giảm chiều dữ liệu.
        Sử dụng hai lớp fully-connected với các hàm kích hoạt ReLU và sigmoid để tạo trọng số.
        Hạn chế: Chi phí tính toán cao và mất thông tin ở mức độ không gian (spatial level).

    - Các cải tiến: 
        GSoP Attention: Sử dụng convolution 1×1 và tính toán sự tương quan kênh.
        ECA Attention: Thay fully-connected layers bằng convolution 1D để giảm chi phí tính toán.

1. Temporal Attention
    - Tập trung vào các thời điểm quan trọng trong dữ liệu tuần tự, ví dụ: Video: Chọn các khung hình chứa thông tin quan trọng để nhận dạng hành động.
    - Temporal Adaptive Module (TAM):
        Có hai nhánh: Nhánh cục bộ (local) và nhánh toàn cục (global).
        Nhánh cục bộ: Sử dụng các convolution 1D để tính trọng số cục bộ.
        Nhánh toàn cục: Sử dụng fully-connected layers để tạo trọng số toàn cục.
        
2. Self-Attention
    - Được sử dụng trong xử lý ngôn ngữ tự nhiên (NLP), lần đầu tiên được đề xuất trong bài toán dịch máy.
    - Cách hoạt động:
        Dữ liệu đầu vào được chuyển đổi thành query, key, và value thông qua linear projection.
        Trọng số được tính bằng dot product giữa query và key, chuẩn hóa và áp dụng softmax.
        Self-Attention là thành phần cơ bản của kiến trúc Transformer.
3. Spatial Attention
    - Tập trung vào các vùng không gian (spatial) quan trọng trong dữ liệu, ví dụ:
        Xác định các vùng trọng tâm trong ảnh để dự đoán chính xác hơn.
    - Attention Gate:
        Sử dụng convolution 1×1 và sigmoid để tạo trọng số cho các vùng không gian.

## 2.3 Activation Functions
Nói về việc sigmoid và hyperbolic tangent (được ưa dùng hơn sigmoid vì gradient mạnh hơn và dễ hội tụ hơn) gây ra vấn đề vanishing gradient trong các mạng deep learning. Do đó nó bị thay thế bởi Relu activation function. Nhiều biến thể của Relu đã được ra đời như Leaky ReLU [163], sigmoid linear unit [167] và  exponential linear unit [39]. Mỗi biến thể đều mang lại những lợi thế riêng để xây dựng các ứng dụng dựa trên học sâu.

## 2.4 Parameter Learning and Loss Functions

Weights (hay parameters) của các mô hình học sâu thường được tối ưu bằng các thuật toán gradient descent. Ngoài gradient descent, còn có rất nhiều các thuật toán tối ưu trọng số khác cũng có thể được sử dụng.

Thuật toán này optimal weights bằng cách cập nhật trọng số theo từng lần lặp training set gọi là epoch sao cho nó sẽ làm cho gía trị loss trở nên tối thiểu trên toàn tập training set. => Để giảm chi phí tính toán nên thực hiện cập nhật theo batch instead.

Tuy nhiên phương pháp train theo batch không đảm bảo bao h cũng hội tụ đến trường hợp tối ưu. Vì nó có thể bị kẹt ở điểm cục bộ hoặc điểm có cùng gradient. 1 số phương pháp đã được thực hiện để cải thiện vấn đề này như cập nhật weight mỗi 1 epoch hay giảm gradient decent theo mini-batch. Cách tiếp cận này tạo ra sự cân bằng, cung cấp gradient ít nhiễu hơn và quy trình đào tạo ổn định hơn. Ngoài ra còn thực hiện adaptive learning để học tự thích ứng learning và momemtune để cải thiện hiệu quả đào tạo và sự hội tụ của các mô hình học sâu.

Về bài toán phân loại, các loss được sử dụng thường là dạng log hoặc cross entropy. Các loss như square loss, absolute loss được sử dụng cho regression problem (phân loại hồi quy)

## 2.5 Regularization Methods

Là các kỹ thuật dùng để ngăn chặn over fitting và cải thiện khả năng tổng quát hóa của mô hình (generalization)

1. Early stopping
    - Theo dõi validate error để phát hiện mô hình bắt đầu overfit, nếu validate error càng ngày càng giảm và training error càng ngày càng tăng thì stop lại.
    - Đặt ra các tiêu chí để dừng vì việc phát hiện điểm khởi đầu của quá trình over fit là 1 thách thức lớn.
2. Dropout
    - Tắt ngẫu nhiên 1 số neutron trong các hiden layers với xác xuất xác định trước trong quá trình tạo mạng neutron.
    - Cần tinh chỉnh dropout rate để cân bằng giữa điều chuẩn và năng lực mô hình (thường nằm trong khoảng 0.1 - 0.8 tùy theo nghiên cứu).
3. Parameter Norm Penalty (Phạt theo chuẩn tham số):
    - Thêm một thuật ngữ phạt (penalty term) liên quan đến trọng số của mạng vào hàm mất mát (loss function).
    - Các thuật ngữ phạt phổ biến bao gồm:
        - L1 norm penalty (giảm số lượng trọng số lớn bằng cách tạo độ thưa thớt).
        - L2 norm penalty (hay còn gọi là weight decay, giảm giá trị trọng số lớn).
        - Kết hợp cả L1 và L2.
4. Batch Normalization và Layer Normalization:
    - Batch Normalization: Chuẩn hóa đầu vào của neuron trên một mini-batch nhằm giảm sự thay đổi trong phân phối dữ liệu sau mỗi lần cập nhật trọng số, giúp tăng tốc độ huấn luyện.hạm vi rộng hơn các tính năng trong khi chiều sâu tạo điều kiện thuận lợi ch
    - Layer Normalization: Chuẩn hóa theo các neuron thay vì mini-batch, phù hợp cho mạng hồi tiếp (Recurrent Neural Networks) và không phụ thuộc vào kích thước mini-batch.


# 3. Type of Deep learning
## 3.1. Deep Supervised Learning (học sâu có giám sát)
Có 3 loại mô hình học sâu là multilayer perceptron, convolutional neural network và recurrent neural network (mạng hồi quy)
1. Multilayer perceptron:
    - Là mô hình gồm các hidden layer là fully connected layer.
    - **Độ rộng (width của mỗi layer), độ sâu (số lượng các hidden layer) ảnh hưởng đến khả năng học của mô hình**
        - Chiều rộng ảnh hưởng đến khả năng nắm bắt các feature của input rộng hơn
        - Chiều sâu ảnh hưởng khả năng phân cấp các feature.
    - Các nhà nghiên cứu đã chỉ ra rằng 1 perceptron nhiều lớp với 1 hidden layer có thể sấp sỉ thành bất kỳ continous function nào.
    - Tuy nhiên, multilayer perceptron yêu cầu input phải là vector 1 chiều. Không phù hợp với dữ liệu phi cấu trúc như hình ảnh, văn bản và giọng nói. Để tận dụng được mạng này, cần phải chuyển đổi feature thành vector 1 chiều, điều này có thể làm mất thông tin cấu trúc không gian của feature.
2. Recurrent Neural Network (RNN):
    - Tận dụng tính tuần tự trong quá trình hình thành và ghi nhớ thông qua việc sử dụng các kết nối đệ quy. Cho phép xử lý dữ liệu hiệu quả như chuỗi thời gian, văn bản, giọng nói và các mẫu tuần tự khác.
    - Việc đào tạo các mạng dạng này gặp phải vấn đề vanishment gradient do thuật toán probation of gradient với long sequence of data. Các phương pháp khắc phục đã được ra đời.
    - Mạng có thể được xây dựng bằng các fully connected và convolution layer.
3. Convolution Neural Network: các kiến trúc nổi bật và giải quyết các thách thức, đặc biệt là trong vấn đề recognition sample và xử lý dữ liệu dạng image.
    - Là các mạng sử dụng CNN layers để bảo toàn và tận dụng thông tin không gian (dài, rộng)cục bộ của input 
    - Kiến trúc điển hình của 1 CNN model bao gồm các CNN layer, Pooling layer đan xen và cuối cùng là fully connected layer. Các lớp CNN và Pooling được dùng để extract các feature nổi bật của đối tượng và sắp xếp phân tầng. Sau cùng được flatten và đưa vào convolution layer để dự đoán output.
    - Trong thập kỷ qua, nhiều mạng CNN đã được đề xuất, trọng tâm là cải tiến là tối ưu khả năng học feature và giải quyết các thách thức như vanishing gradient và giảm việc tái sử dụng feature
    
    - Alexnet (2012) là mô hình CNN đầu tiên được công nhận rộng rãi và đánh dấu mốc đáng kể trong lĩnh vực deep learning for computer vision.
        - Gồm 5 lớp tích chập và các Pooling layer thực hiện sau lớp tích chập 1 và 2, tiếp theo là 3 lớp fully connected.
        - Các filter sử dụng là 11 x 11, 5 x 5, 3 x 3
        - Relu được sử dụng để giảm thiểu vanishing gradient.

    - VGG-16: Cải tiến bằng cách thêm nhiều lớp hơn, chỉ sử dụng các bộ lọc 3 x 3.
        - Nhấn mạnh về tầm quan trọng độ sâu của mạng, việc chỉ sử dụng các filter 3 x 3 giảm khả năng.

    - ZFNet:
        - Kiến trúc hệt Alexnet chỉ khác dùng các filter và stride nhỏ hơn và chuẩn hóa feature maps cho phép mô hình capture các features tốt hơn và cải thiện hiệu suất tổng thể.
    
    - Network-in-Network (NIN): Giới thiệu
        - Hai phép tích chập 1×1 sau phép tích chập k×k, mô phỏng như một multilayer perceptron trên bản đồ đặc trưng, giúp trích xuất đặc trưng trừu tượng hơn.
        - Thay vì flatten feaure ở layer cuối cùng, NIN model tính toán spatial average của mỗi feature map và kết quả cuối cùng được đưa vào softmax để phân loại. => Giảm đáng kể tham số phải train trong fully connected layer cuối cùng.
    
    - **GoogleNet (Inception)**
        - Giới thiệu module "inception", kết hợp các bộ lọc kích thước 5×5, 3×3, và 1×1 trong các pipeline song song để trích xuất thông tin ở các mức độ chi tiết khác nhau.
        - Module này được xếp chồng lên nhau ở các lớp cao hơn, trong khi các lớp tích chập truyền thống và max-pooling được sử dụng ở các lớp thấp hơn.
        - Áp dụng phép pooling trung bình toàn cầu (global average pooling) trước lớp fully-connected cuối cùng để giảm số lượng tham số.

        ![](images/2.%20Inception%20layer.png)

    **Các cải tiến giải quyết vần đề gradient vanishment**
    - Highway Network:
        - Giới thiệu cơ chế gating để điều chỉnh luồng thông tin giữa các lớp, cho phép thông tin từ các lớp trước truyền qua các lớp sau một cách hiệu quả, giải quyết vấn đề độ dốc biến mất và giúp đào tạo mạng nơ-ron rất sâu (lên đến 100 lớp).
    
    - Resnet(Residual Network)):
        - Sử dụng kết nối dư (residual connection), cho phép thông tin bỏ qua một số lớp, giúp giải quyết vấn đề độ dốc biến mất mà không cần thêm tham số.
        - ResNet tích hợp các khối dư (residual block) bao gồm 2 hoặc 3 lớp tích chập với chuẩn hóa batch (batch normalization) và ReLU.
            ![](images/2.%20Residual%20connection.png)
        - Kết nối dư không bao giờ đóng, đảm bảo mọi thông tin luôn được truyền qua các lớp, giúp đào tạo mạng rất sâu (tối đa 152 lớp).
            - Độ sâu ảnh hưởng đến khả năng học: Khi số lượng lớp tăng lên, ngay cả với skip connections, các lớp ở đầu có thể trở nên không hữu ích (do diminishing feature reuse ở phần sau) và việc học của mô hình có thể tập trung chủ yếu ở các lớp gần cuối, làm giảm hiệu quả của lớp đầu.
            - Tăng chi phí tính toán: Mỗi layer thêm vào làm tăng đang kể chi phí tính toán 
            - Khi mở rộng đến 152 lớp, chi phí tính toán đã gần đạt giới hạn thực tế trên phần cứng tại thời điểm ResNet được phát triển.
            - Sự bão hòa trong cải thiện hiệu suất: Các nghiên cứu cho thấy, sau một số lượng lớp nhất định (ví dụ: 152 lớp), việc tăng thêm độ sâu không mang lại cải thiện đáng kể về hiệu suất, trong khi chi phí tính toán lại tăng mạnh.
            - Khả năng mất ổn định trong rất sâu: Dù skip connections giúp giảm thiểu vanishing gradient, việc đào tạo một mạng quá sâu (hơn 152 lớp) vẫn có thể gặp khó khăn trong việc hội tụ do gradient explosion hoặc các vấn đề tối ưu hóa phi tuyến tính khác.
    - DenseNet:
        - Mở rộng ý tưởng của ResNet và HighWay Network bằng cách giới thiệu "kết nối dày đặc" (dense connection).
        - Mỗi lớp tích chập nhận đầu vào là tất cả các feature map từ các lớp trước đó bằng skip connection, tối đa hóa luồng thông tin.
        - Giới thiệu Dense block như hình dưới

            ![](images/2.%20Dense%20block.png)
        
            - Bao gồm nhiều CNN layer, mỗi CNN layer sử dụng batch normalization, Relu activation function và 3 x 3 convolution.
            - Có thể thấy Dense block khá tốn kém do số lượng các feature map ngày càng tăng.. Để giảm chi phí tính toán, các khối chuyển tiếp (transition block) được thêm vào. Bản chất mỗi khối là tích chập 1×1 và max-pooling được thêm vào để giảm số chiều không gian (spatial dimention) của feature map.
            
        - Độ sâu của mạng có thể đạt tối đa 264 layers.
        
    **Các cải tiến giải quyết vấn đề diminishing feature reuse**
    Mặc dù skip connections làm giảm vấn đề vanishing gradient, tuy nhiên 1 thách thức mới lại xuất hiện. Đó là việc tái sử dụng feature ở các class sâu hơn (do skip connection đưa feature vào các layer sâu hơn 1 cách trực tiếp). Diminishing feature reuse đề cập vấn đề giảm hiệu quả của các feature map của các layer trước đó tác động lên kết quả predict cuối cùng.
    - WideResnet: 
        - Là 1 kiến trúc CNN dựa trên resnet với mục đích giảm thiểu vấn đề diminishing feature reuse. 
        - Thay vì làm mạng sâu hơn, nó mở rộng chiều rộng của mỗi layer (channels) theo hệ số k. Việc mở rộng channel giúp model học được các feature đa dạng hơn.
    - Resnext:
        - Giải quyết vấn đề diminishing feature reuse bằng cách tăng khả capture các feature và đa dạng feature hơn từ input.
        - Giới thiệu khái niệm cardinally dựa theo inception module của google net
        
        ![](images/2.%20Cardinal%20block.png)

## 3.2 Deep Unsupervised Learning (học sâu không giám sát, dataset không có label)
Các mô hình này thường dựa vào tối ưu hóa hàm mục tiêu nào đó để nắm bắt các đặc điểm cơ bản như phân cụm các sample, học các feature embedding có ích và tái tạo input data từ các biểu diễn nén.

Ví dụ về các mô hình học không giám sát như autoencodes, generative adversarial networks và restricted Boltzmann machines.

1. Restricted Boltzman machines: 
    - Là generative neural network model, dùng để học .....\

# 4. Applications of Deep Learning
## 4.1. Computer Vision
### 4.1.1 Image Classification

Image Classification là tasks cơ bản trong lĩnh vực computer vision, cần phải phân loại các ảnh thành 1 trong các lớp xác định trước đó dựa trên nội dung của ảnh. Các mạng đã được sử dụng trong lĩnh vực này.

1. LeNet-5: được giới thiệu là mạng CNN đầu tiên phân loại thành công loại chữ số viết tay.
2. AlexNet: 2012
3. ZFNet, VGG16, GoogleNet, ResNet và ResNext: Chú trọng vào tăng độ sâu của mạng, giải quyết các vấn đề vanishing gradient và diminishing of feature reuse.
4. Các nghiên cứu về phân loại hình ảnh tiếp tục được phát triển với trọng tâm là nâng cao hiệu suất mô hình phân loại. 1 xu hướng đang là trend hiện này là xây dựng loss function để bỏ qua các trường hợp được phân loại tốt và phân phối mất cân bằng của class.
    - Cross-entropy loss: công thức này khuyến khích các mô hình chú ý đến các mẫu phân loại tốt và tệ.
    - Asymmetric polynomial loss: cho phép quá trình train ưu tiên có chọn lọc các đóng góp của các trường hợp tích cực để giảm thiểu mất cân bằng giữa các lớp tiêu cực và tích cực. Nó đòi hỏi phải tinh chỉnh 1 số lượng lớn các parameter và có thể dẫn đến over fitting
    - 
5. Kết hợp nhiều mô hình deeplearning để cải thiện hiệu suất tổng thể. Tuy nhiên việc này gặp nhiều khó khăn do số lượng tham số cần train lớn.
    - 1 cách giảm thiểu tham số cần train là sử dụng quy tắc trọng số.

6. ViT - Vison transform: 1 giải pháp thay thế mạng CNN vốn đã thống trị trong lĩnh vực phân loại ảnh.
    - Tận dụng các cơ chế self-attention để có thể học representation learning trên quy mô lớn, khác biệt với CNN khi không tập trung vào các local feature.
    - **ViT nhạy cảm với việc điều chỉnh hyperparameter và hiệu suất trên các dataset nhỏ dưới mực tiêu chuẩn, thiếu khả năng tận dụng các đặc trưng không gian trong khi mạng CNN làm được.**
    - Do đó 1 số research cố gắng kết hợp CNN và ViT để cải thiện hiệu suất và khả năng hoạt động.
        ![](images/2.ViT.png)
        - conformer: 
            - Gồm 1 nhánh CNN để trích xuất đặc trưng cục bộ (local CNN) và 1 nhánh transformer global feature tương ứng. 
            - 2 nhành này được kết nối với nhau bởi 2 'bridges' là 1 x 1 covolution và up/down sampling operator. Output của 2 nhánh được tổng hợp lại để tạo ra output cuối cùng.
        - MaxViT:
            - Cũng có kiến trúc kết hợp CNN và ViT, giải quyết vấn đề thiếu khả năng mở rộng (scalability) của cơ chế self-attention.
            - Có 2 module:
                - Module 1: Xử lý local feature từ các patches không overlapping.
                - Module 2: Xử lý global feature từ các grid 
### 4.2.1. Object Detection
Các mạng deeplearning cải thiện đáng kể hiệu xuất trong lĩnh vực Object detection. Điểm khác lĩnh vực này so với Classification.
- Xác định số lượng đối tượng xuất hiện, bounding box của từng đối tượng. Trong khi classification sẽ gán nhãn cho toàn ảnh.
- Tính toán phức tạp hơn do phải xử lý thêm thông tin về vị trí và kích thước đối tượng.

1. R-CNN:
    - Là bước đột phá đầu tiên trong phát hiện đối tượng kết hợp CNN với bouding box (region proposals - vùng đề xuất, đóng vai trò là các ứng viên tiềm năng cho output sau cùng).
    - Cơ chế hoạt động
        - Region Proposals: Sinh các vùng (bounding box) khả thi trong ảnh (vùng có thể chứa đối tượng).
        - Feature Extraction: Sử dụng CNN để trích xuất đặc trưng từ các vùng này.
        - Classification: Phân loại các vùng để xác định đối tượng.
    - Hạn chế: Rất chậm và tốn tài nguyên.

2. Fast R-CNN
    - Cải tiến từ R-CNN
    - Cơ chế hoạt động: gồm 2 nhánh dự đoán
        - Phân loại đối tượng (object classification).
        - Xác định vị trí chính xác hơn (bounding box regression).
    - Ưu điểm: Hiệu suất tốt hơn.
    - Nhược điểm: Vẫn chậm và không phù hợp cho ứng dụng thời gian thực.

3. Faster R-CNN: tích hợp thêm Region Proposals (RPN) vào Fast-CNN để tăng tốc độ và hiệu suất
    - Region Proposal Network (RPN)
        - Sinh các vùng đề xuất (bounding box) trực tiếp từ ảnh đầu vào.
        - Mỗi vùng đi kèm với một confidence score để biểu thị mức độ tin cậy của việc chứa đối tượng.
        - Sử dụng **Anchor Boxes**: Định nghĩa các bounding box với tỷ lệ khung hình khác nhau trên feature maps do CNN tạo ra.
        - Các anchor boxes được hiệu chỉnh (regressed) để định vị đối tượng chính xác.
    - Quy trình huấn luyện:
        - Bước 1: Huấn luyện RPN để sinh các vùng đề xuất.
        - Bước 2: Sử dụng các vùng này để huấn luyện Fast R-CNN cho phát hiện đối tượng.
    - Mạng backbone được sử dụng để extract features với Faster R-CNN là ZFNet hoặc VGG16.
        ![](images/3.%20Faster%20R-CNN.png)
    - Code Faster R-CNN với backbone là ResNet50: https://viblo.asia/p/trien-khai-faster-rcnn-cho-cac-bai-toan-detection-OeVKBMoE5kW
            ![](images/3.%20Faster%20R-CNN-1.png)
        - RPN function tạo layer RPN để đề xuất các anchor box: 
            Đầu vào của nó là feature map extract được từ 1 mạng backbone nào đó như Resnet50, kích thước tùy ý.
            Lớp CNN đầu tiên có tác dụng tạo ra các anchor boxes từ feature map ban đầu. Nó là các bounding box với các tỷ lệ khác nhau Đầu ra lớp này phụ thuộc vào kích thước feature map đầu vào và có 52 kênh.
            **Lớp tiếp theo là object_score chứa 9 Conv 1x1. Lý do chọn 9 filter là do trong RPN, mỗi điểm trên feature map sẽ tạo ra 9 anchor boxes (tương ứng với các tỷ lệ và tỉ lệ khung hình khác nhau). Sigmoid Activation: Được sử dụng để xác định xác suất (probability) của mỗi anchor box có chứa đối tượng hay không. Kết quả đầu ra sẽ là một ma trận với số lượng anchor boxes, mỗi anchor box có một giá trị xác suất thuộc khoảng [0, 1], cho biết "objectness score" (xác suất có đối tượng trong anchor box đó). Output: objectness_score có kích thước (height, width, 9), trong đó height và width phụ thuộc vào kích thước của feature_map đầu vào. 9 là số lượng anchor boxes tại mỗi điểm trên feature map.**
            **Layer tiếp theo là lớp hồi quy Bounding Box với input đầu vào là feature từ rpn_layer. Đây là 1 lớp Conv 1x1 với 36 filter. Mỗi bỗ lọc này sẽ dự đoán các điều chỉnh cho các bounding boxes. Vì mỗi bounding boxes cần có 4 thông số (x, y, width, height) nên 9 anchor boxes cần 36 gía trị. Linear activation được sử dụng để trả về gía trị liên tục cho việc điều chỉnh các bounding boxes. Output: bbox_regression có kích thước (height, width, 36), với mỗi điểm trên bản đồ đặc trưng có 36 giá trị điều chỉnh cho các anchor boxes tương ứng.**
            Output sau cùng: 
                objectness_score: Một tensor có kích thước (height, width, 9), mỗi giá trị trong tensor này đại diện cho xác suất có đối tượng trong một anchor box tại vị trí đó.
                bbox_regression: Một tensor có kích thước (height, width, 36), mỗi giá trị trong tensor này là một giá trị điều chỉnh (regression) cho các bounding boxes tại vị trí đó
            - RPN tạo ra các region proposals (vùng đề xuất) dưới dạng các anchor boxes.
            - Mỗi điểm trên feature map được liên kết với 9 anchor boxes, mỗi box sẽ có một xác suất cho việc chứa đối tượng (objectness score) và một bộ điều chỉnh cho kích thước và vị trí của bounding box.
            - objectness_score cho biết độ tin cậy của anchor box, trong khi bbox_regression cung cấp thông tin để điều chỉnh bounding box sao cho chính xác hơn.
            ![](images/3.%20Faster%20R-CNN-2.png)
        - RoI Pooling: Chuẩn hóa các vùng đề xuất về kích thước cố định.
        - Detection Head: Phân loại đối tượng và điều chỉnh các bounding boxes dựa trên các đặc trưng của các vùng đề xuất.

4. Yolo - You Only Look Once
    - Các mạng object detectors trên được chia làm 2 stage, trong đó stage đầu tiên generate ra các region proposals sau đó mới detect object. Nhược điểm của các mạng này là cần tài nguyên tính toán lớn và không phù hợp với các ứng dụng realtime.
    - Yolo đề xuất giải pháp chỉ có 1 stage duy nhất bằng cách trực tiếp detect bounding box và object's confidence score trong 1 lần lan truyền duy nhất. Hiển nhiên phù hợp với các ứng dụng thời gian thực.
    - Input image được chia thành grid SxS, mỗi grid có trách nhiệm detect object trong grid đó. Cụ thể với mỗi grid, hàng loạt các bounding box được dự đoán với confidence core cao, cho phép phát hiện đối tượng đồng thời trên toàn bộ image. 
    - Các version sau của Yolo chủ yếu tập trung vào việc cải thiện hiệu suất mô hình.

5. Single Shot Multibox Detect (SSD)
    - Là 1 mạng detect bounding box 1 stage bên cạnh Yolo, cũng nhằm giải quyết vấn đề detect realtime.
    - Nó cũng trực tiếp detect bounding box, confidence scores và giảm phức tạp khi tính toán.
    - SSD tạo ra các dự đoán ở các cấp độ khác nhau của feature map, cho phép model detect object ở các size khác nhau của input.

Vấn đề tỷ lệ không cân đối giữa foreground và background. Trong bài toán object detection, thường có sự mất cân đối lớn giữa foreground (vùng đối tượng cần phát hiện) và background (vùng không chứa đối tượng). Ví dụ như trong một ảnh lớn, chỉ có một phần nhỏ là đối tượng cần phát hiện, còn lại là nền. Điều này khiến mô hình dễ tập trung vào việc phân loại đúng các background (vì chiếm đa số), nhưng bỏ qua các foreground quan trọng.

6. RetinaNet
    - Giải quyết vấn đề trên bằng focal loss dựa trên cross entropy. Ý tưởng chính: Focal Loss giảm trọng số của các mẫu dễ phân loại (như background) để mô hình tập trung hơn vào các mẫu khó phân loại (như các đối tượng foreground nhỏ hoặc không rõ ràng).
    - Áp dụng FPN (Feature Pyramid Network) với Resnet làm backbone để trích xuất các feature.
    - FPN là một kiến trúc có cấu trúc hình kim tự tháp, giúp mô hình:
        - Nắm bắt được các đặc trưng đa tỉ lệ (multiscale features) từ ảnh.
        - Phát hiện đối tượng với kích thước khác nhau (nhỏ, trung bình, lớn) hiệu quả hơn.

7. EfficientDet
    -  Cải tiến từ FPN bằng cách sử dụng Bi-directional Feature Pyramid Network (Bi-FPN). Bi-FPN thực hiện multi-level feature fusion (hợp nhất đặc trưng từ nhiều tầng) để mô hình nắm bắt đặc trưng đa tỉ lệ hiệu quả hơn.
    - EfficientNet được sử dụng làm backbone network thay vì ResNet.
        - EfficientNet là một mạng tiên tiến, được thiết kế để tối ưu hóa giữa hiệu quả tính toán (computational efficiency) và độ chính xác (accuracy).

Hiệu suất object detect thường dựa vào bước xử lý cuối cùng gọi là NMS - non-maximum suppression. Mục đích của lớp này là để loại bỏ các bounding box trùng lặp, chồng lên nhau sau cùng. Cụ thể, nó sẽ sắp xếp các bounding box có confidence score, chọn 1 box có score cao nhất và loại bỏ các box có score thấp hơn mà trùng đáng kể với box này. 

Tuy nhiên điều này gặp phải vấn đề về sự không nhất quán giữa score và chất lượng của bounding box. Cụ thể NMS vẫn giữ lại các bounding box được định vị kém với điểm số tin cậy cao trong khi loại bỏ các dự đoán chính xác hơn với điểm số tin cậy thấp. Soft-NMS được giới thiệu để giảm thiểu vấn đề này, soft-NMS áp dụng Gausian lên các box đó thay vì loại bỏ chúng để giảm confidence scores. Ý tưởng không phải là loại bỏ các bounding box lân cận, mà
giảm dần điểm số của chúng dựa trên mức độ chồng chéo với bounding box đã chọn. Điều này dẫn đến một sự triệt tiêu mượt mà hơn, bảo toàn các hộp giới hạn được định vị tốt hơn.

Adaptive NMS giới thiệu cơ chế tự thích ứng threshold để loại bỏ các bounding box overlap. Thuật toán này sẽ tự động điều chỉnh threshold dựa trên mức độ overlap với selected box.


8. Detection Transformer (DeTr): Object detection sử dụng Transform. Nó loại bỏ các thành phần thủ công như anchor box, 
..

### 4.1.3. Image Segmentation
Các reseache thay thế fully connected layer bằng các conv 1x1 sau đó bilinear up-sampling để match size của input, đây là các mạng khởi đầu của vấn đề image segmentation.

1. Deconvolution network:
    - Là 1 mạng khác để semantic segmentation.

### 4.1.4. Image Generation
0. Chia làm 3  giai đoạn chính.
    - Extract Feature từ input text.
    - Tạo ảnh dựa trên thông tin đã trích xuất được
    - Kiểm soát quá trình tạo ảnh, đảm bảo đầu ra đáp ứng các tiêu chí hoặc ràng buộc cụ thể.

1. Đoạn này tập trung vào bước 2
    - Variational Autoencoder (VAE):
        - Một trong những mô hình đầu tiên có khả năng tạo ảnh.
        - Cách hoạt động:
            - VAE học cách sinh dữ liệu bằng cách nắm bắt phân phối tiềm ẩn (Gaussian) của dữ liệu huấn luyện.
            - Trong quá trình tạo ảnh, các tham số phân phối được lấy mẫu và đưa vào decoder để tạo ảnh.
        - Hạn chế:
            - Hình ảnh tạo ra thường bị mờ và chưa đạt chất lượng cao.
        - Ý nghĩa:
            - Mặc dù còn hạn chế, VAE đã chứng minh tiềm năng của học sâu trong việc tạo ảnh.
    - Generative Adversarial Networks (GANs):
        - Được giới thiệu sau VAE, GAN đã cải thiện đáng kể chất lượng ảnh được tạo.
        - Cách hoạt động:
            - Gồm hai mạng nơ-ron liên kết:
                - Generator: Học cách tạo ảnh chân thực để đánh lừa discriminator.
                - Discriminator: Học cách phân biệt giữa ảnh thật và ảnh giả.
            - Hai mạng này được huấn luyện đồng thời theo cách cạnh tranh.
        - Ưu điểm: Ảnh tạo ra ít bị mờ hơn và trông chân thực hơn.
        - Các cải tiến trên GAN:
            - CGAN (Conditional GAN): Cho phép chỉ định đặc điểm của hình ảnh sẽ được tạo (ví dụ: màu sắc, đối tượng, ...).
            - DCGAN (Deep Convolutional GAN):
                - Sử dụng kiến trúc convolutional để cải thiện sự ổn định trong quá trình huấn luyện.
                - Đặt nền tảng cho nhiều cải tiến khác trong GANs.
1. 
    - StackGAN:
        - Phân chia quy trình tạo ảnh thành hai giai đoạn:
            - Stage-I: Tạo ảnh độ phân giải thấp với bố cục cơ bản (hình dạng, màu sắc, và nền) từ vector nhiễu ngẫu nhiên.
            - Stage-II: Bổ sung chi tiết để tạo ra ảnh có độ phân giải cao, trông chân thực hơn.
        - StackGAN++ (phiên bản nâng cao của StackGAN):
            - Gồm nhiều bộ sinh ảnh (generators) với tham số được chia sẻ.
            - Các bộ sinh trung gian tạo ra ảnh ở các kích thước khác nhau, và bộ sinh sâu nhất tạo ra ảnh có chất lượng cao nhất.
            - Cải thiện khả năng sinh ảnh đa tỉ lệ (multiscale).

    - HDGAN (Hierarchical Discriminative Generative Adversarial Network):
        - Đặc điểm nổi bật:
            - Bộ sinh đơn luồng (single-stream generator) kết hợp với các bộ phân biệt (discriminators) phân tầng.
            - Các tầng trung gian sinh ra ảnh ở nhiều độ phân giải khác nhau.
        - Lợi ích:
            - Ảnh có độ phân giải thấp dùng để học cấu trúc ngữ nghĩa (semantic structure).
            - Ảnh có độ phân giải cao dùng để học các chi tiết tinh vi (fine-grained details).

    StackGAN và chất lượng đầu ra giai đoạn đầu (Stage-I): StackGAN phụ thuộc nhiều vào chất lượng ảnh sinh ra ở Stage-I.
    - Cải tiến với DM-GAN:
        - Memory Network: Được tích hợp để cải thiện ảnh chất lượng thấp từ Stage-I.
        - Chức năng:
            - Lựa chọn các từ liên quan từ dữ liệu đầu vào.
            - Tinh chỉnh chi tiết ảnh để nâng cao chất lượng và tạo ra ảnh chân thực hơn.
    
    Các mô hình như StackGAN, StackGAN++, và HDGAN tập trung vào việc tạo ảnh đa tỉ lệ, với các bộ sinh ảnh và bộ phân biệt làm việc ở các cấp độ khác nhau để cải thiện chất lượng tổng thể. DM-GAN đi xa hơn bằng cách sử dụng một mạng trí nhớ để xử lý ảnh chất lượng thấp ở giai đoạn đầu, giúp tối ưu hóa chất lượng đầu ra ở cấp độ cao hơn.

2. 