# Reflectance Models | Radiometry and Reflectance
## Mục tiêu và vấn đề
Ở bài trước ta đã biết được các đặc tính phản xạ của mọi vật liệu có thể được biểu diễn bởi hàm BRDF. Bây giờ ta sẽ xem xét 1 số BRDF/Reflectance models phổ biến được áp dựn trong thực tiễn. Cụ thể là hàm f chính xác là gì.
## Các cơ chế phản xạ

1. Specular Reflection (phản xạ gương)

    ![Specular reflection](images/5.%20Specular%20Reflection.png)

    Các vật thể có vẻ ngoài bỏng bẩy, sáng bóng và chủ yếu là các bề mặt nhẵn như gương, kính và các loại được đánh bỏng khác

2. Body Reflection 

    ![Body reflection](images/5.%20Body%20Reflection.png)

    Phản xạ đi 1 phần vào vật thể. Bản thân các vật liệu thường có cấu tạo không đồng nhất được tạo bởi các hạt khác nhau nên khi ánh sáng đi vào sẽ bị khúc xạ và phản xạ rất nhiều lần và có thể quay trở lại bề mặt. Thường sự phản xạ nà xảy ra ở không sâu so với bề mặt vật liệu và rất phức tạp. Khi tia phản xạ quay trở lại surface có thể đi tán xạ theo hướng bất kỳ.

    Body reflection: là vì tia chiếu sáng của nguồn đi vào bên trong bề mặt vật chất. Ngoài ra còn được gọi là Diffuse reflection (phản xạ khuếch tán), mang lại cho các vật thể 1 vẻ ngoài mờ (matte apperance) do bản chất không đồng nhất của vật thể.

    Ví dụ: đất xét, giấy, các loại chất lỏng, ...

    Vì vậy Image Intesity không chỉ là Surface Reflectance thông thường mà còn thêm cả Body Reflectance.

3. Example

    ![](images/5.%20Example.png)

    Cái chậu cây đất xét chủ yếu là Body Reflection => Matte apperance (vẻ ngoài mờ)

    Còn quả cầu là specular reflection (phản xạ gương), không thực sự có tia sáng nào đi vào bên trong bề mặt vật thể.

    Thực tế còn có trường hợp giao thoa cả 2 mô hình phản xạ này như ở trường hợp cuối cùng, có 1 phần trong hình ảnh bị lóe sáng là sự kết hợp của 2 mô hình phản xạ. Bạn có thể thấy được body của nó cũng như khả năng gương của nó.

## Các mô hình phản xạ

1. Lambertian (áp dụng với phản chiếu loại body effect)

    ![Lambert - BRDF](images/5.%20Lamber%20-%20BRDF.png)

    Surface có độ sáng như nhau mọi hướng. Bất kể bạn theo dõi từ hướng nào nó vẫn có cùng độ sáng bề mặt. Nói cách khác BRDF của nó là hằng số.

    Theo công thức trên hình d-diffuse (khuếch tán), p tương trưng cho albedo (hiệu suất phản chiếu) có giá trị từ 0 đến 1 (0 là không phản chiếu gì cả, vật đen xì, 1 là vật có màu trắng - bề mặt phản chiếu tất cả ánh sáng mà nó nhận được)

    Mô hình này mô tả gần đúng rất nhiều loại bề mặt và vì tính đơn giản của mô hình phản xạ này nên nó được sử dụng rất nhiều trong lĩnh vực computer vison và graphics.

    Mối quan hệ giữa surface radiance (L) và Image Irradiance

    ![Lambert model](images/5.%20Lamber%20Model.png)

    Cuối cùng ta thu được công thức cuối như hình.
        J: là Radiant Intensity (thông lượng ánh sáng nguồn phát ra trong 1 đơn vị góc soild)
        Pd: là hằng số phụ thuộc vào bề mặt, tính chất vật liệu
        n: vector đơn vị của pháp tuyến bề mặt
        s: vector đơn vị hướng chiếu sáng của nguồn.

    Nhìn vào công thức trên ta sẽ thấy r^2. Điều gì sẽ xảy ra nếu ta di chuyển nguồn sáng xung quang bề mặt lambert mà vẫn giữ khoảng cách r ?
    
    ![Larbert surface](images/5.%20Lambert%20Surface.png)

    Nếu chiếu nguồn sáng từ trên xuống => theta i = 0, khi đó n.s =1. Lúc này có thể thấy tia phản xạ bằng nhau mọi hướng
    Và nếu tăng theta i, các tia phản xạ vẫn bằng nhau mọi hướng tuy nhiên độ sáng sẽ giảm so với lúc đầu.=> Đây là định nghĩa mô hình phản xạ lambert

    Mô hình lambert thực sự đại diện cho 1 đầu của quang phổ, là sự phản chiếu lý tưởng của body reflection. Bất kể sự chiếu sâng của nguồn từ đâu, bề mặt luôn có vẻ sáng như nhau từ mọi hướng

2. Ideal Specular model (surface - áp đụng cho các bề mặt phản chiếu loại gương)

    ![Ideal specular](images/5.%20Ideal%20Specular.png)

    Phản xạ gương: bề mặt sẽ phản xạ tất cả ánh sáng mà nó nhận được mà không đi qua bề mặt vật thể. Toàn bộ ánh sáng sẽ bị phản xạ 1 hướng duy nhất

    Giả sử bạn chiếu sáng từu hướng s thì nó sẽ phản xạ tất cả theo 1 hướng duy nhất là r. r gọi là specular direction. Người quan sát chỉ nhận được ánh sáng từ surface nếu v=r.

3. Example

    ![](images/5.%20Example%202.png)

    Đầu tiên là ví dụ về mô hình lambert (chí có thể áp dụng với body reflection)

        Giả sử bạn đang nhìn quả cầu có bề mặt lambert, hướng nhìn là v, nguồn sáng là s, ... và nguồn sáng, camera đặt ở vô cực, góc giữa chúng là theta.

        Trong hình ảnh mà camera tạo ra, bạn có thể thấy điểm sáng nhất của quả cầu là p (giao của s và quả cầu nơi mà soild angle = 0). Theo lý thuyết ở trên, khi nguồn sáng vuông góc với vật thể thuộc body reflection thì lượng ánh sáng mà nó phản chiếu là lớn nhất.

        Ngoài ra các điểm thuộc 1 đường tròn (hình quả cầu) sẽ có độ sáng như nhau bởi vì góc của mỗi điểm này đến nguồn bằng nhau.

    Tiếp theo là mô hình ideal specular

        Trong cả hình cầu, chỉ có điểm q duy nhất làm cho v=r. Máy ảnh chỉ nhận được lượng ánh sáng phát ra từ điểm này nên trong ảnh, bạn sẽ chỉ nhìn thấy duy nhất điểm q mà không thấy các điểm khác trên quả cầu (hình trên họ vẽ thêm tượng trưng quả cầu thôi, chứ chính ra hình ảnh chỉ có duy nhất 1 điểm sáng là q)