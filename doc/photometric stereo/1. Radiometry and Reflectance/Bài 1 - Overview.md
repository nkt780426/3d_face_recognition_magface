# Nguồn: https://www.youtube.com/playlist?list=PL2zRqk16wsdpyQNZ6WFlGQtDICpzzQ925

# Overview | Radiometry and Reflectance (đo bức xạ và tương phản)

## Vấn đề: 
Điều kiện: Có 1 bức ảnh (image) được chụp từ scene (môi trường được dùng để chụp ảnh) và tại 1 pixel nào đó trong ảnh đo được brightness (độ sáng) là 65. 

Vấn đề: Vậy từ số 65 máy tính có thể hiểu được gì từ scene ? Rõ ràng máy tính không thể hiểu hết được scene do có quá nhiều yếu tố có thể ảnh hưởng đến chất lượng ảnh chụp như độ sáng nguồn sáng, bề mặt chụp, ... 

Tính nghiêm trọng: Vấn đề này rất nghiêm trọng khi huấn luyện 1 mô hình vì máy tính không hiểu được trọn vẹn scene chỉ từ 1 bức ảnh dẫn đến nhưng feature mà model extract được sẽ bị hạn chế đi rất nhiều => **Vấn đề này gọi là Image Intensity Understanding.**

Để giải quyết vấn đề cần biết 2 điều: 
    radiometric concept (các khái niệm đo bức xạ): là những khái niệm liên quan đến độ sáng như brighness of source/scene, ...
    reflectance (sự phản chiếu) : Khả năng của 1 vật thể / bề mặt (surface) có thể tán xạ ray từ light source và các reflectance models

Thuật ngữ:
    direction: hướng của ray
    reflect: phản xạ
    scene: môi trường thực tế trong ảnh
    image: ảnh được chụp bởi camera - nó là 1 ma trận numpy/tensor/...
    Image Intensity (cường độ điểm ảnh): là 1 số trng ma trận ảnh. Image Intensity Understand là khả năng ta có thể hiểu được scene của ảnh như thế nào trong môi trường thực tê.
    ray: tia sáng từ nguồn
    illuminate(động từ - cái gì đó chiếu sáng): ví dụ source illuminate surface, đôi lúc động từ này được dịch thành ánh sáng
    Irradiance: Nói về lượng ánh sáng nhận được trên 1 đơn vị diện tích,
    Rough surface: bề mặt gồ ghề

## Reflectance models (phản xạ)
1. Giả sử có 1 hệ thống thị giác máy tính như sau: Máy tính nhận image được tạo ra bởi camera và hiển thị với người dùng.

    ![vision system](./images/1.%20typical%20vision%20system.png)

    - Scenes: là không gian trong cuộc sống, nơi chụp ảnh.
    - Source of lighting: các nguồn sáng được chia làm 2 loại chính là point of light (điểm sáng) và extended source (nguồn sáng mở rộng - tức là tập hợp các điểm như bóng đèn ở gần nơi chụp, ... - bản chất vẫn đưa về point of light). Hiển nhiên khi chụp ảnh scene có thể có 1 hoặc nhiều nguồn sáng, mỗi nguồi sáng có thể là điểm hoặc nguồn sáng mở rộng. Hình trên mô tả trường hợp đơn giản nhất là scene có 1 nguồn sáng và là nguồn sáng điểm.

2. Cách 1 điểm trong scene được biểu diễn bởi camera tạo ra ảnh.

    ![pixel](./images/1.%20single%20point%20of%20the%20scene.png)

    Tại 1 điểm bất kỳ trong scene. Nó nhận ray nguồn sáng và reflect (phản xạ) hay scatters (tán xạ) ra các direction (hướng) khác nhau. Trong các tia tán xạ đó, có 1 tia đi vào thấu kính máy ảnh (len hay còn gọi là sensor). Tia đó là tia tới trong vật lý, qua thấu kính sẽ thu được tia phản xạ.

    Có 1 màn chắn image plane nằm sau thấu kính camera và hứng tia sáng này. Độ sáng của nó chính là giá trị image intensity.

3. Nhắc lại vấn đề: Từ Image Itensity thu được trên Image plane, ta/máy tính có thể rút ra được đặc trưng nào của scene. Để đơn giản hóa vấn đề ta sẽ đưa vấn đề về điểm trên image tương ứng với điểm trong scene.

## Đi sâu hơn vào vấn đề.

0. Ở phần trên ta đã có cái nhìn tổng quan về cách 1 Image Intensity được đo, phần này chỉ thêm cái pháp tuyến n và lưu ý thêm về bề mặt vật liệu mà ta chụp sẽ tán xạ nguồn sáng. Mỗi 1 vật liệu có khả năng tán xạ tia sáng khác nhau (chiết suất n) và nó là 1 hằng số với mỗi loại vật liệu.
    
    ![Image Intensity](./images/1.%20Image%20Intensity%20Understanding.png)

1. Các nhân tố ảnh hưởng đến việc tính toán có thể kể đến như
    Illumination (ánh sáng): là ánh sáng điểm hay mở rộng, có bao nhiêu nguồn sáng trong scene và độ sáng nó thể nào, hướng mà nó chiếu sáng đến điểm trên surface mà ta đang xét, ... Các nhân tố này có thể biểu diễn bởi rất nhiều biễn x.
    Surface Orientation (định hướng bề mặt): tức là vector pháp tuyến n của bề mặt tại điểm ta đang xét. Nó ảnh hưởng đến tia tán xạ, phản xạ.
    Surface Reflectance (Khả năng phản xạ của bề mặt): mỗi 1 bề mặt sẽ có 1 hẳng số phản xạ riêng. Mà 1 vật thể có thể được tạo bởi nhiều vật liệu nên để biểu diễn nó ta cần rất nhiều tham số

2. Từ các nhân tố trên, camera sẽ tìm cách ánh xạ nó qua hàm f nào đó để thu được Image Intense. Từ đây ta có thể thấy rõ ràng vấn đề Image Intense Understand của máy tính: máy tính chỉ biết được image intensity (1 biến ở vế trái), nếu muốn nó hiểu được scene của ảnh sẽ phải xác định toàn bộ các biến x ở vế phải. Điều này gần như là không thể dù ta biết được hàm f. Vấn đề Image Intensity Understand bị hạn chế nghiêm trọng, các mô hình học máy sẽ không bao giờ có thể extract toàn bộ các đặc trưng của vật thể thông qua image do bản thân dataset không chứa đủ thông tin cần thiết.

3. **Giải pháp: hóa ra vấn đề Image Intensity có thể được giải quyết nếu ta chụp ảnh trong 1 số điều kiện thích hợp. Ví dụ từ hàm f(x) ở trên**
	Nếu ta chụp ảnh trong môi trường đã biết tất cả nguồn sáng như độ sáng nguồn, hướng chiếu, nguồn sáng điểm hay mở rộng, ... và vật thể được chụp là mặt người (biết chụp gì sẽ biết được hang số reflectance của vật liệu)
	Như vậy ở hàm trên. Ta đã biết tất hầu như tất cả các tham số: vế trái là Image Intensity (cường độ điểm ảnh), vế phải thì biết tất cả trừ tham số Surface Orientation (vector pháp tuyến của điểm) và có thể dễ dàng tính được
	Với mỗi điểm trên ảnh ta làm như vậy, kết quả ta sẽ thu được 1 ma trận vector pháp tuyến của cả vật thể. Ma trận này được gọi là normal map của vật thể. Từ normal map ta có thể dễ dàng dựng hình ảnh 3D của vật thể.
	Phương pháp này là photometric stereo và chỉ là 1 trong số rất nhiều các phương pháp có thể tái tạo ảnh 3D.

## Tổng kết.
Tổng kết: Để thực hiện Image Intensity, ta cần phải chụp ảnh trong 1 điều kiện nhất định nào đó. Tuy nhiên ta sẽ bàn về nó chi tiết trong từng phương pháp tái tạo 3D. Trước mắt ta cần hiểu 2 khái niệm Radiometry concepts (các khái niệm bức xạ - liên quan đến độ sáng của nguồn, hướng sáng, ...) và Reflectance properties (khả năng phản xạ của vật thể).

1. Radiometric Concepts: Các khái niệm liên quan đến khả năng chiếu sáng của nguồn lên bề mặt, ... Rất nhiều các khái niệm cần phải biết.

2. Surface radiance (độ sáng bề mặt) và Image Irradiance: được sử dụng để đo độ sáng của 1 điểm trong scene
	Surface radiance (độ sáng bề mặt): 1 điểm trong scene sẽ có 1 độ sáng nhất định khi đã được nguồn chiếu sáng. Giá trị đó là surface radiance và phải quy ước 1 unit đo nó.
	Image Irradiance: Khi đã biết surface radiance của 1 điêm trong scene thì tham chiếu ánh xạ nó như thế nào để thành điểm trong ảnh hay độ sáng của điểm trong ảnh là bảo nhiêu khi biết surface radiance ? Để tính được nó ta cần biết về mối quan hệ giữa Image Irradiance (độ sáng của điểm trong ảnh) và Surface radiance (độ sáng của điểm trong scene)
	Tất nhiên ta có thể tưởng tượng rằng nếu Surface radiance tăng (đặ vật gần nguồn sáng hơn, tăng độ sáng nguồn, ...) thì Image Irradiance thu được có thể tăng nhưng nó tăng theo cấp số nhân hay đồ thị parabol, ...

3. BRDF: Bidirectional Reflectance Distribution Function (hàm phân bổ phản xạ
	Sau khi đã biết được các khái niệm trên ta sẽ tìm hiểu về reflectance (khả năng phản xạ của bề mặt vật thể). Có thể hiểu nguồn sáng có tính chất hạt (mang năng lượng) và tia. Khi ray chiếu tới surface tại 1 điểm, nó bị tán xạ nhiều hướng cùng với tán xạ năng lượng. Khi ta đứng tại 1 nơi (camera), mắt ta chỉ thu được 1 số tia sáng trong số các tia tán xạ và có năng lượng (độ sáng) khác nhau.
	BRDF là 1 hàm mô tả mức độ các tia tán xạ theo các hướng quan sát khác nhau tùy thuộc vào tính chất bề mặt (nhẵn, mờ hoặc phản xạ) từ đó ta có thể mô phỏng cách ánh sáng tương tác với bề mặt vật thể từ đó tạo ra ảnh 3D chân thực hay phân tích bề mặt của đối tượng để hiểu rõ hơn về vật liệu của chúng.
	Cụ thể BRDF định nghĩa tỷ lệ ánh sáng phản xạ từ 1 bề mặt khi ray tới và quan sát từ 1 hướng khác (camera)
		Đầu vào: hướng chiếu sáng của nguồn đến bề mặt (so với vector pháp tuyến n), hướng quan sát, độ sáng nguồn, ...
	Từ hàm này, ta rút ra được 1 vài thuộc tính reflectance (phản xạ/tán xạ) thường gặp trong cuộc sống và điều đó giúp ta tạo ra các mô hình phản xạ (reflectance models)

4. Reflectance Models
	Ta sẽ xem xét các bề mặt đặc biệt và 1 số mô hình tán xạ ánh sáng được sử dụng với chúng trong lĩnh vực computer vison và computer graphics

5. Dichroma models
	Sau khi đã xem xét được cường độ ánh sáng của các tia phản xạ, ta sẽ xem xet reflectance (độ phản xạ) của 1 điểm bị ảnh hưởng như thế nào bởi màu sắc của điểm đó (bề mặt màu đen, đỏ, ...) => làm thế nào để màu sắc của nguồn sáng bị sửa đổi bởi bề mặt tạo thành tia tán xạ có tính chất như thế nào => Biết được nó ta có thể biết được màu sắc của vật thể trong scene và tái tạo nó thành màu trong ảnh.
	Dichromatic models (mô hình phản xạ lưỡng sắc) cách để chụp 1 image duy nhất và tách nó ra thành các thành phần phản xạ của nó (reflectance)