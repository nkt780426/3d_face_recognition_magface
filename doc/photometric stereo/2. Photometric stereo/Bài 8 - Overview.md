# Overview | Photometric Stereo
## Giới thiệu
Ở Phần 1 ta đã hiểu về các khái niệm cơ bản về reflectance (phản xạ), mô tả độ sáng của nguồn, bề mặt, lượng ánh sáng mà camera thu được từ 1 điểm, các mô hình phản xạ, lưỡng sắc, ... Với tất cả những điều này ta đã đủ kiến thức để đến với phương pháp đầu tiên giải quyết vấn đề Image Intensity Understand => Photometric Stereo

Photometric Stereo (âm thanh nổi trắc quang) là phương pháp khôi phục thông tin 3 chiều từ các intensity value trong ảnh 2D (recovering three-dimensional shape information) trong 1 số điều kiện đặc biệt, nơi ta đã biết hầu hết các thông tin về scene.

## Image Intensity

Cùng nhìn lại các nhân tố ảnh hưởng đến việc tái tạo Image Intensity của ảnh.

![](images/8.%20Image%20Intense%20Understand.png)

Camera ở hướng v, nguồn sáng hướng s và pháp tuyến bề mặt tại điểm quan sát là n (cả 3 vector trên đều là unit vector). Ngoài ra hướng xem v còn được biểu thị bởi trục z (không có gì đảm bảo s, v, n cùng nằm trên 1 mặt phẳng trong không gian 3 chiều).

Image Intensity (cường độ điểm ảnh) là y = f(X), trong đó X là ma trận các nhân tố ảnh hưởng đến việc tính toán image intensity như trên hình và hàm f .
- source: được xác định bởi các tham số như vector s, light flux, radient intensity, khoảng cách của nó đến bề mặt.
- Surface normal: môi 1 điểm trên bề mặt có 1 vector pháp tuyến riêng. Nếu ta tính toán được vector pháp tuyến của tất cả mọi điểm trên bề mặt ta sẽ tái tạo được ảnh 3D. Ngoài ra còn 1 số tham số khác
- surface reflectance propertites: surface irradiance, surface radiance, bề mặt vật liệu, reflectance models, rough surface (bề mặt gồ ghề)

Và như hình vẽ, nếu ta chụp ảnh trong 1 điều kiện nhất định (đã biết tất cả trừ vector pháp tuyến bề mặt) thì từ image intensity, ta/máy tính có thể tính được surface normal
- Giả sử rằng đó là nguồn sáng điểm và ở khoảng cách rất xa so với surface => Tất cả mọi điểm trên surface đều nhìn nguồn sáng theo hướng vector s.
- Các đặc tính phản xạ của bề mặt đều biết ví dụ như mặt người thường sẽ có 1 đặc tính bề mặt giống nhau.

## Photometric sterro (âm thanh nổi trắc quang)

**Photometric stereo là phương pháp tính toán, ước tính (estimating) các surface normal tại từng điểm trên surface dựa trên các đặc tính phản xạ bề mặt (surface reflectance) và nguồn sáng đã biết. Và thực tế để biết được những đặc tính phản xạ của bề mặt ta chỉ cần nhiều hình ảnh chụp với nhiều vị trí nguồn sáng khác nhau, sau đó chồng những ảnh này lên là có thể tính toán surface orientation (hướng của bề mặt) của mỗi điểm.**

![](images/8.%20Photometric%20stereo.png)

1. Gradiant space và reflectance map:
    Gradiant space là một không gian được định nghĩa (sẽ học sau) như không gian Oxyz, RGB, ... từ đó giúp ta có 1 cái gì đó để chuẩn hóa các tham số. Nhờ gradiance space, ta có thể dễ dàng biểu diễn orientation, vector v và surface normal tại mỗi điểm trên bề mặt.

    Reflectance map: cung cấp cho bạn cách mapping giữa surface orientaion/surface gradient (độ dốc bề mặt) và intensity của image. Là 1 công cụ hữu ích (toàn gọi là normal map)

2. Photometric stereo

    Từ refleactance map chúng ta có thể phát triển photometric stereo.
    Vấn đề: ta có 1 image intensity được đo bởi 1 nguồn sáng duy nhất thì sẽ có vô số pháp tuyến bề mặt (surface normal) hay độ dốc bề mặt (surface gradients) sẽ tạo ra cùng 1 image intensity.
    Giải pháp: để giải quyết vấn đề này ta sử dụng multi light source => Chỉ cần 1 lượng nhỏ nguồn sáng là ta có thể xác định được pháp tuyến tại mỗi điểm của bề mặt.

3. Calibration base photometric stereo
    
    Có những trường hợp ta có thể biết được biểu thức reflectance model của surface dưới dạng BRDF đơn giản. Ví dụ như Lambertiam  model.

    Tuy nhiên có rất nhiều trường hợp ta biêt mình đang xử lý cái gì, vật liệu mà chúng ta sẽ xử lý mà không biêt được BRDF của nó. Trong trường hợp này ta phát triển phương pháp (calibration base photometric stereo - phương pháp dựa trên hiệu chuẩn). Chúng ta sẽ lấy màu sơn mà ta đã biết và sơn vật thể đã biết đó. Và sau đó sử dụng đối tượng hiểu chỉnh đó callbration object để tạo ra bảng looc up (tra cứu) cho phép bạn ánh xạ nó với image intensity mà bạn đo được và map chúng với surface orientation hoặc surface gradients.

4. Shape from surface normal
    Sau khi áp dụng phương pháp photometric stereo, thứ bạn thu được sau cùng là surface normal của vật thể. Nếu surface này là continous (liên tục) thì chúng ta có thể tích hợp các surface normal để khôi phục hình dạng 3D của vật thể. Chúng ta sẽ xem xét các cách tiếp cận khác nhau để có thể làm được điều này.

